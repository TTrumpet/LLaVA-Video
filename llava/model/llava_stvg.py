# TODO: implement other base models
import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from .language_model.llava_qwen import LlavaQwenForCausalLM 

# Import custom modules
from .stvg_modules import DifferentiableGatingSelector, RLBudgetPolicy, BBoxDecoder

from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast

class LlavaQwenForSTVG(LlavaQwenForCausalLM):
    """
    LlavaQwen model extended for Spatio-Temporal Video Grounding (STVG).

    This model incorporates:
    1. Adaptive Token Selection to filter visual tokens.
    2. A Hypernetwork-based BBox Decoder for continuous trajectory prediction.
    """
    def __init__(self, config):
        super().__init__(config)

        # New components
        self.model.token_selector = DifferentiableGatingSelector(token_dim=config.hidden_size)
        self.model.bbox_decoder = BBoxDecoder(latent_dim=config.hidden_size)
        self.model.budget_policy = RLBudgetPolicy(state_dim=config.hidden_size, min_k=64, max_k=4096) # Example values

        # Optional: Add a hyperparameter for balancing the losses
        self.bbox_loss_lambda = getattr(config, 'bbox_loss_lambda', 1.0)

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, k_values=None):
        #print("In prepare input labels for multimodal")
        torch.set_printoptions(profile="full")
        #torch.set_printoptions(profile="default")
        #print(input_ids)
        #print(len(input_ids[0]))

        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
                
        else:
            image_features = self.encode_images(images)
        
        # --- STVG token selection --- #
        # 1. Get the token budget (K) for each item in the batch.
        batch_size = len(image_features)
        #k_values = self.budget_policy.get_budget(batch_size)
    
        # 2. Iterate through each feature tensor and select the top K tokens.
        selected_image_features = []
        for i, feature_tensor in enumerate(image_features):
            # The selector expects a batch dimension, so we add and remove it.
            # Shape goes from (num_tokens, dim) -> (1, num_tokens, dim)
            feature_tensor_batched = feature_tensor.unsqueeze(0)
            
            # Get the specific K for this item
            k_for_item = k_values[i].unsqueeze(0) # Shape: (1,)
    
            # Apply the selector
            selected_tokens = self.model.token_selector(feature_tensor_batched, k_for_item)
            
            # Remove padding and the batch dimension before appending.
            # The selector pads to max(K), so slice to the actual K for this item.
            num_to_keep = int(k_for_item.item())
            selected_tokens_unpadded = selected_tokens[:, :num_to_keep, :]
            
            selected_image_features.append(selected_tokens_unpadded.squeeze(0))
    
        # 3. Replace the original features with new, filtered features.
        image_features = selected_image_features

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        #rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):

            # extract bboxes from input_ids
            # assuming one bbox start and end token
            '''
            if torch.is_tensor(cur_input_ids):
                input_ids_as_list = cur_input_ids.tolist()
            else:
                input_ids_as_list = cur_input_ids

            try:
                bbox_start = input_ids_as_list.index(BBOX_INDEX)
                bbox_end = input_ids_as_list.index(BBOX_INDEX, bbox_start+1)
                bboxes = input_ids_as_list[bbox_start+1:bbox_end] # bboxes as a single list of normalized integers out of 1000

                #bbox = get embedding from trained model

                # remove bboxes and replace with ignore token
                input_list_without_bbox = input_ids_as_list[:bbox_start]
                #input_list_without_bbox.extend([IGNORE_INDEX] * (len(bboxes)+2))
                #input_list_without_bbox.append(IGNORE_INDEX)
                input_list_without_bbox.extend(input_ids_as_list[bbox_end+1:])
                #print(input_list_without_bbox)

                cur_input_ids = torch.tensor([input_list_without_bbox], dtype=torch.long, device=cur_input_ids.device)
            except Exception as e:
                print(e)
                # no bboxes in input
                print("No bboxes detected.")

            print(cur_input_ids)
            '''

            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            #rank0_print(num_images)
            # no images, no bboxes
            '''
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            '''

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

            # replace image and bbox tokens
            '''
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            bbox_token_indices = torch.where(cur_input_ids == BBOX_INDEX)[0].tolist()
            token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + torch.where(cur_input_ids == BBOX_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            token_indices.sort()
            '''
            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            #rank0_print(image_token_indices)
            #rank0_print(bbox_token_indices)

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            # image and bbox replace 
            '''
            bboxes = None
            for i in range(len(token_indices) - 1):

                # if bbox start token, do not append
                # if bbox_token_indices is empty, there are no bbox tokens in the query
                if token_indices[i] == bbox_token_indices[0]:
                    bboxes = cur_input_ids[token_indices[i] + 1 : token_indices[i + 1]]
                    #print("Extracted bboxes: ", bboxes)
                else:
                    cur_input_ids_noim.append(cur_input_ids[token_indices[i] + 1 : token_indices[i + 1]])
                    cur_labels_noim.append(cur_labels[token_indices[i] + 1 : token_indices[i + 1]])

            cur_bboxes = bboxes
            '''

            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            #for index in token_indices:
            #    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            #    cur_new_labels.append(cur_labels_noim[i])

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))


            '''
            added_bbox_embedding = False
            for i in range(len(split_sizes)):
                #print("length of new input embeds: ", len(cur_new_input_embeds))
                #print("shape of current non-image embedding: ", cur_input_embeds_no_im[i].shape)
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                #print("i: ", i)
                #print("token_indices[i + 1]: ", token_indices[i+1])
                if token_indices[i + 1] in image_token_indices:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    #print("shape of current image embedding: ", cur_image_features.shape)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                # adding bbox embedding
                elif token_indices[i + 1] in bbox_token_indices:
                    if not added_bbox_embedding:
                        bbox_features = self.encode_bboxes(cur_bboxes)
                        #print("shape of bbox embedding: ", bbox_features.shape)
                        cur_new_input_embeds.append(bbox_features)
                        # trying to get the model to output BBOX_INDEX in place of spatial info
                        cur_new_labels.append(torch.full((bbox_features.shape[0],), BBOX_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                        added_bbox_embedding = True
                    else:
                        pass
            '''

            #print(len(cur_new_input_embeds))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        #rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # TODO: inserting bbox embeddings
        #rank0_print("Inserting bbox embeddings")

        #rank0_print("Finish preparing")
        #print(new_input_embeds[0].shape)
        #print(new_input_embeds)
        #exit()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False, # Kept for compatibility
        # --- NEW ARGUMENTS --- #
        timestamps: Optional[torch.Tensor] = None,
        ground_truth_bboxes: Optional[torch.Tensor] = None,
        bbox_attention_mask: Optional[torch.Tensor] = None,
        tubelet_token_indices: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Always request hidden states and a dictionary output for our logic
        output_hidden_states = True
        return_dict = True
        
        # --- RL POLICY --- #
        # 1. Create a state from the input text.
        # We create a simple sentence embedding by averaging the token embeddings.
        # Note: We mask out padding tokens before averaging.
        text_embeds = self.get_model().embed_tokens(input_ids)
        # Create a mask for non-padding tokens
        text_mask = (input_ids != 0).unsqueeze(-1).to(text_embeds.dtype)
        # Calculate the mean of embeddings, avoiding division by zero
        state = (text_embeds * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
        # --- RL POLICY END --- #
    
        # 2. Call the policy to get the budget K and log_probs.
        k_values, log_probs = self.model.budget_policy(state)

        # 1. Prepare inputs (text + selected visual tokens)
        # This will automatically call your *overridden* prepare method
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, k_values=k_values,
            )

        # 2. Pass inputs through the base model (standard forward pass)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 3. Perform BBox Decoding
        bbox_predictions = None
        loss_bbox = None
        reward = None
        
        if tubelet_token_indices is not None:
            # Extract the latent vector 'z' from the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            z = torch.gather(last_hidden_state, 1, tubelet_token_indices.view(-1, 1, 1).expand(-1, -1, last_hidden_state.size(-1))).squeeze(1)

            # Decode the bounding boxes
            if timestamps is not None:
                bbox_predictions = self.model.bbox_decoder(z, timestamps)

                # Calculate regression loss if ground truth is provided
                if ground_truth_bboxes is not None and bbox_attention_mask is not None:
                
                    # Select real predictions > corresponding real gt boxes
                    valid_predictions = bbox_predictions[bbox_attention_mask]
                    valid_gt_bboxes = ground_truth_bboxes[bbox_attention_mask]
                
                    bbox_loss_fn = nn.L1Loss() # Or GIoU, etc.
                    loss_bbox = bbox_loss_fn(bbox_predictions, ground_truth_bboxes)
                    
                    with torch.no_grad(): # Use no_grad for reward calculation
                        # Reward = Accuracy - Cost. A simple accuracy proxy is (1 - loss).
                        accuracy_proxy = 1.0 - loss_bbox
                        
                        # Penalize using a larger K to encourage efficiency
                        cost_penalty = 0.001 # This is a hyperparameter you can tune
                        computation_cost = cost_penalty * k_values.float()
                        
                        # Final reward for each item in the batch
                        reward = accuracy_proxy - computation_cost

        # 4. Combine losses and prepare final output
        policy_loss = None
        if reward is not None:
            # The REINFORCE loss. The minus sign is because optimizers minimize.
            # We want to MAXIMIZE (log_prob * reward).
            policy_loss = -torch.mean(log_probs * reward)
        
        total_loss = outputs.loss
        if loss_bbox is not None and torch.is_tensor(total_loss):
            total_loss += self.bbox_loss_lambda * loss_bbox
        
        if policy_loss is not None and torch.is_tensor(total_loss):
            # Add a new lambda for the policy loss to tune its influence
            total_loss += self.policy_loss_lambda * policy_loss

        return STVGOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # Your custom values:
            loss_bbox=loss_bbox,
            policy_loss=policy_loss,
        )
        
        
@dataclass
class STVGOutput(CausalLMOutputWithPast):
    # Add fields for your custom losses
    loss_bbox: torch.FloatTensor = None
    policy_loss: torch.FloatTensor = None