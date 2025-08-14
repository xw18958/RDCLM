import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import random

def get_class_lookup_table():
    return {
    "NIDC_knowledge_base":0,
    "IDC_knowledge_base":1,
    "benign_colon_knowledge_base":0,
    "malignant_colon_knowledge_base":1,
    "benign_breasKHis":0,
    "malignant_breaKHis":1,
    "benign_gpt_knowledge_base":0,
    "malignant_gpt_knowledge_base":1,
}

def get_chosen_knowldge_class(llm_kn):
    if llm_kn=="gpt":
        return [
        "benign_gpt_knowledge_base",
        "malignant_gpt_knowledge_base"
        ]
    elif llm_kn=="gemini":
        return [
        "NIDC_knowledge_base",
        "IDC_knowledge_base",
        "benign_breasKHis",
        "malignant_breaKHis",
        ]
    else:
        raise ValueError(f"Unknown llm_kn: {llm_kn}. Supported values are 'gpt' and 'gemini'.")

# def get_chosen_knowldge_class(llm_kn):
#     if llm_kn=="gpt":
#         return [
#         "benign_gpt_knowledge_base",
#         "malignant_gpt_knowledge_base"
#         ]
#     elif llm_kn=="gemini":
#         return [
#         "NIDC_knowledge_base",
#         "IDC_knowledge_base",
#         "benign_breasKHis",
#         "malignant_breaKHis",
#         ]
#     else:
#         raise ValueError(f"Unknown llm_kn: {llm_kn}. Supported values are 'gpt' and 'gemini'.")

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)): 
        init.kaiming_normal_(m.weight)

def unit_vector(v):
    # m = np.sqrt(np.sum(v*v))
    m = torch.sqrt(torch.sum(v*v))
    return v/m

def compute_acc(labels,preds):
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    num_correct = torch.sum(labels==preds)
    num_l = len(labels)
    return num_correct/num_l

def cos_sim(t_emb,i_emb):
    u_t = unit_vector(t_emb)
    u_i = unit_vector(i_emb)
    # print(f"u_i:{u_i.shape}")
    cos_s = u_t @ u_i.T
    return cos_s.T
    

def get_fused_features(list_f):
    f_sum = 0
    for f in list_f:
        f_sum+=f
    f_fused = unit_vector(f_sum)
    return f_fused

def get_index_retrieved_top_k_text(emb_knowledge_base,i_emb,k):
    cos_sim_i_knowledge_base = cos_sim(emb_knowledge_base,i_emb)
    # print(cos_sim_i_knowledge_base.shape)
    top_k_values, top_k_indices = torch.topk(cos_sim_i_knowledge_base, k=k, dim=1)
    # print(top_k_values.shape)
    # print(top_k_indices.shape)
    return top_k_indices

def get_retrieved_top_k_text(knowledge_base,emb_knowledge_base,i_emb,k):
    index_retrieved_top_k_text = get_index_retrieved_top_k_text(emb_knowledge_base,i_emb,k)
    index_retrieved_top_k_text = index_retrieved_top_k_text.cpu()
    # print(f"index_retrieved_top_k_text:{index_retrieved_top_k_text.shape}")
    # print(f"index_retrieved_top_k_text:{index_retrieved_top_k_text}")
    # if index_retrieved_top_k_text.is_cuda:
    #     print("Tensor is on GPU")
    # else:
    #     print("Tensor is on CPU")
    return knowledge_base[index_retrieved_top_k_text]

def generate_projected_img_emb(vlmodel,image_encoder,processor,img,device):
    img_inputs = processor(images=img, return_tensors="pt", padding=True, do_rescale=False)
    img_pixels = img_inputs.pixel_values
    img_pixels = img_pixels.to(device)
    image_encoder.eval()
    with torch.no_grad():
        # img_output = image_encoder(img_pixels,output_hidden_states=True)
        img_output = image_encoder(img_pixels)
        
    # vision_final_state = img_output.hidden_states[-1]
    vision_final_state = img_output.last_hidden_state
    # print(vision_final_state.shape)

    projected_img_emb = vlmodel(vision_final_state)
    # vision_out = vision_out.to(device)
    # projected_img_emb = projected_img_emb.unsqueeze(1)
    return projected_img_emb

def generate_mutual_projected_img_emb(vlmodel,image_encoder,processor,img,device):
    img_inputs = processor(images=img, return_tensors="pt", padding=True, do_rescale=False)
    img_pixels = img_inputs.pixel_values
    img_pixels = img_pixels.to(device)
    image_encoder.eval()
    with torch.no_grad():
        # img_output = image_encoder(img_pixels,output_hidden_states=True)
        img_output = image_encoder(img_pixels)
        
    # vision_final_state = img_output.hidden_states[-1]
    vision_final_state = img_output.last_hidden_state
    # print(vision_final_state.shape)

    projected_img_emb,branch_embs = vlmodel(vision_final_state)
    # vision_out = vision_out.to(device)
    # projected_img_emb = projected_img_emb.unsqueeze(1)
    return projected_img_emb,branch_embs

def id_to_emb(ids,model,device):
    # print("ids:",ids.is_cuda)
    ids = ids.to(device)
    emb = model.get_input_embeddings()(ids)
    return emb

def get_text_ids(text,tokenizer,pad_config,max_length):
    text_ids = tokenizer(
    text,
    padding=pad_config,
    # padding="longest",
    # padding="max_length",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
    add_special_tokens=False
    ).input_ids
    return text_ids

def text_to_emb(text,tokenizer,pad_config,max_length,model,device):
    ids = get_text_ids(text,tokenizer,pad_config,max_length)
    # print(f"ids:{ids}")
    emb = id_to_emb(ids,model,device)
    return emb

#get the merged retrieved text and filtered relevant text of the batch
def get_merged_retrieved_and_relevant_text(retrieved_top_k_text,class_lookup_table,ref_lookup_table,b_label):
    ### merge text in a row into a single sentence
    merged_batch_text = []
    b_relevant_text = []
    # print(f"retrieved_top_k_text:{len(retrieved_top_k_text)}")
    for i,list_t in enumerate(retrieved_top_k_text):
        # merged_batch_text.append(" ".join(list_t))
        # print(f"list_t: {list_t}")
        merged_text = "# "+ " # ".join(list_t)
        # merged_batch_text.append(" # ".join(list_t))
        merged_batch_text.append(merged_text)
        # print(f"merged_batch_text: {merged_batch_text}")
        # sys.exit()
        relevant_text = []
        for t in list_t:
            if class_lookup_table[ref_lookup_table[t]]==b_label[i]:
                relevant_text.append(t)
        ###convert from np.str_ to str and merge into a single sentence
        # relevant_text = " ".join([str(t) for t in relevant_text])
        relevant_text = "# "+ " # ".join([str(t) for t in relevant_text])
        b_relevant_text.append(relevant_text)
        # print(f"b_relevant_text:{b_relevant_text}")
        # sys.exit()
    return merged_batch_text,b_relevant_text

#for performance robustness on noise
def get_merged_retrieved_and_relevant_text2(retrieved_top_k_text,class_lookup_table,ref_lookup_table,b_label):
    ### merge text in a row into a single sentence
    merged_batch_text = []
    b_relevant_text = []
    b_noise_levels = []
    for i,list_t in enumerate(retrieved_top_k_text):
        # merged_batch_text.append(" ".join(list_t))
        # print(f"list_t: {list_t}")
        merged_text = "# "+ " # ".join(list_t)
        # merged_batch_text.append(" # ".join(list_t))
        merged_batch_text.append(merged_text)
        # print(f"merged_batch_text: {merged_batch_text}")
        # sys.exit()
        relevant_text = []
        k = len(list_t)
        for t in list_t:
            if class_lookup_table[ref_lookup_table[t]]==b_label[i]:
                relevant_text.append(t)
        ###convert from np.str_ to str and merge into a single sentence
        # relevant_text = " ".join([str(t) for t in relevant_text])
        b_noise_levels.append((k-len(relevant_text))/k)
        relevant_text = "# "+ " # ".join([str(t) for t in relevant_text])
        b_relevant_text.append(relevant_text)
        # print(f"b_relevant_text:{b_relevant_text}")
        # sys.exit()
    return merged_batch_text,b_relevant_text,b_noise_levels

def get_randomized_retrieved_and_relevant_text(retri_random,retrieved_top_k_text,class_lookup_table,ref_lookup_table,b_label):
    ### merge text in a row into a single sentence
    merged_batch_text = []
    b_relevant_text = []
    for i,list_t in enumerate(retrieved_top_k_text):
        #randomize the order of the text
        # print(f"list_t: {list_t}")
        if retri_random:
            random.shuffle(list_t)
        # print(f"list_t: {list_t}")
        merged_text = "# "+ " # ".join(list_t)
        merged_batch_text.append(merged_text)
        relevant_text = []
        for t in list_t:
            if class_lookup_table[ref_lookup_table[t]]==b_label[i]:
                relevant_text.append(t)
        relevant_text = "# "+ " # ".join([str(t) for t in relevant_text])
        # print(f"relevant_text: {relevant_text}")
        b_relevant_text.append(relevant_text)
    return merged_batch_text,b_relevant_text

def get_randomized_retrieved_text_add_noise(random_noise_rate,retri_random,
                                            benign_knowledge_base,
                                            malignant_knowledge_base,retrieved_top_k_text,
                                            class_lookup_table,ref_lookup_table,b_label):
    merged_batch_text = []
    b_relevant_text = []
    for i,list_t in enumerate(retrieved_top_k_text):
        relevant_text = []
        label_i = b_label[i]#label of the ith image in the batch
        ###randomly shuffle the text
        if retri_random:
            random.shuffle(list_t)

        for j,t in enumerate(list_t):
            if class_lookup_table[ref_lookup_table[t]]!=label_i:# if text doesn't match the image label
                if random.random() < random_noise_rate:
                    #replace retreieved irrelevant text with other random irrelevant text from the knowledge base
                    if label_i==0:#image is benign with label 0, get noisy text from counter malignant text
                        ##replace list_t[j] with the  noised irrelevant text
                        list_t[j] = np.random.choice(malignant_knowledge_base,1)[0]
                    else:
                        list_t[j] = np.random.choice(benign_knowledge_base,1)[0]
            #else t == label_i, simply append relevant text
            else:
                relevant_text.append(t)

        merged_text = "# "+ " # ".join(list_t)
        merged_batch_text.append(merged_text)
        relevant_text = "# "+ " # ".join([str(t) for t in relevant_text])
        # b_relevant_text.append(relevant_text + '</s>')#add end of sentence token
        b_relevant_text.append(relevant_text)
    return merged_batch_text,b_relevant_text

def get_randomized_retri_text_add_noise_dual_CLM(random_noise_rate,retri_random,
                                            benign_knowledge_base,
                                            malignant_knowledge_base,retrieved_top_k_text,
                                            class_lookup_table,ref_lookup_table,b_label):
    merged_batch_text = []
    b_relevant_text = []
    b_irrelevant_text = []
    for i,list_t in enumerate(retrieved_top_k_text):
        relevant_text = []
        irrelevant_text = []
        label_i = b_label[i]#label of the ith image in the batch
        ###randomly shuffle the text
        if retri_random:
            random.shuffle(list_t)

        for j,t in enumerate(list_t):
            if class_lookup_table[ref_lookup_table[t]]!=label_i:# if text doesn't match the image label
                if random.random() < random_noise_rate:
                    #replace retreieved irrelevant text with other random irrelevant text from the knowledge base
                    if label_i==0:#image is benign with label 0, get noisy text from counter malignant text
                        ##replace list_t[j] with the  noised irrelevant text
                        irr_t = np.random.choice(malignant_knowledge_base,1)[0]
                        list_t[j] = irr_t
                        irrelevant_text.append(irr_t)
                    else:
                        irr_t = np.random.choice(benign_knowledge_base,1)[0]
                        list_t[j] = irr_t
                        irrelevant_text.append(irr_t)
            #else t == label_i, simply append relevant text
            else:
                relevant_text.append(t)

        merged_text = "# "+ " # ".join(list_t)
        merged_batch_text.append(merged_text)
        relevant_text = "# "+ " # ".join([str(t) for t in relevant_text])
        irrelevant_text = "# "+ " # ".join([str(t) for t in irrelevant_text])
        # b_relevant_text.append(relevant_text + '</s>')#add end of sentence token
        b_relevant_text.append(relevant_text)
        b_irrelevant_text.append(irrelevant_text)
    return merged_batch_text,b_relevant_text,b_irrelevant_text

def get_randomized_retrieved_text_add_noise_and_instruc_noise(instruc_rate,random_noise_rate,retri_random,
                                            benign_knowledge_base,
                                            malignant_knowledge_base,retrieved_top_k_text,
                                            class_lookup_table,ref_lookup_table,b_label):
    merged_batch_text = []
    b_target_text = []
    instruc_inticator = []
    b_text_labels = []
    for i,list_t in enumerate(retrieved_top_k_text):
        target_text = []
        label_i = b_label[i]#label of the ith image in the batch
        ###randomly shuffle the text
        if retri_random:
            random.shuffle(list_t)

        instruc_rand_v = random.random()
        do_irr_instruc = instruc_rand_v < instruc_rate
        text_labels = []
        for j,t in enumerate(list_t):
            if class_lookup_table[ref_lookup_table[t]]!=label_i:# if text doesn't match the image label
                # text_labels.append(int(not label_i))
                if random.random() < random_noise_rate:
                    #replace retreieved irrelevant text with other random irrelevant text from the knowledge base
                    if label_i==0:#image is benign with label 0, get noisy text from counter malignant text
                        ##replace list_t[j] with the  noised irrelevant text
                        list_t[j] = np.random.choice(malignant_knowledge_base,1)[0]
                    else:
                        list_t[j] = np.random.choice(benign_knowledge_base,1)[0]

                if do_irr_instruc:
                    target_text.append(list_t[j])
                    # continue
            #else t == label_i, simply append relevant text
            else:
                # text_labels.append(int(label_i))
                if not do_irr_instruc:
                    target_text.append(t)
        # print(f"instruc_rand_v,instruc_rate: {instruc_rand_v,instruc_rate}")
        b_text_labels.append(text_labels)
        if do_irr_instruc:
            instruc_inticator.append("irr")
        else:
            instruc_inticator.append("rel")
        merged_text = "# "+ " # ".join(list_t)
        merged_batch_text.append(merged_text)
        target_text = "# "+ " # ".join([str(t) for t in target_text])
        b_target_text.append(target_text)
    # return merged_batch_text,b_target_text,instruc_inticator,b_text_labels
    return merged_batch_text,b_target_text,instruc_inticator

def get_label_ids(target_text,tokenizer,max_length):
    # encode the targets
    label_ids = tokenizer(
        target_text,
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    # replace padding token id's of the labels by -100 so it's ignored by the loss
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    return label_ids

def construct_multimodal_sentence_emb(prefix1_emb,proj_v_emb,prefix2_emb,retrieved_text_emb,len_sentence):#descriptions
    b_size = proj_v_emb.shape[0]
    prefix1_emb = prefix1_emb.repeat(b_size,1,1)
    prefix2_emb = prefix2_emb.repeat(b_size,1,1)
    multimodal_sentence_emb = torch.cat([prefix1_emb,proj_v_emb],dim=-2)
    multimodal_sentence_emb = torch.cat([multimodal_sentence_emb,prefix2_emb],dim=-2)
    multimodal_sentence_emb = torch.cat([multimodal_sentence_emb,retrieved_text_emb],dim=-2)
    return multimodal_sentence_emb[:,:len_sentence,:]

def construct_multimodal_sentence_emb_with_diff_instruc(prefixes_embs,proj_v_emb,prefix2_emb,retrieved_text_emb,len_sentence):#descriptions
    multim_sent_embs = []
    for i,prefix in enumerate(prefixes_embs):
        multimodal_sentence_emb = torch.cat([prefix,proj_v_emb[i].unsqueeze(0)],dim=-2)
        multimodal_sentence_emb = torch.cat([multimodal_sentence_emb,prefix2_emb],dim=-2)
        multimodal_sentence_emb = torch.cat([multimodal_sentence_emb,retrieved_text_emb[i].unsqueeze(0)],dim=-2)
        # print(f"multimodal_sentence_emb: {multimodal_sentence_emb.shape}")
        multimodal_sentence_emb = multimodal_sentence_emb[:,:len_sentence,:]
        # print(f"multimodal_sentence_emb: {multimodal_sentence_emb.shape}")
        multim_sent_embs.append(multimodal_sentence_emb)
    return torch.stack(multim_sent_embs,dim=1).squeeze(0)

def generate_sentence_mask(b_size,prefix1_id,prefix2_id,i_token,text_mask):
    len_mask1 = prefix1_id.shape[1]+i_token.shape[1]+prefix2_id.shape[1]
    mask1 = torch.ones(b_size,len_mask1)
    concat_mask = torch.cat([mask1,text_mask],dim=1)
    #get the size of the text mask to be the mask size
    final_mask = concat_mask[:,:text_mask.shape[1]]# for padding = max_length
    # final_mask = concat_mask# modify for padding = longest
    # print(f"final_mask:{final_mask.shape}")
    return final_mask

def generate_sentence_mask_with_diff_instruc(prefixes_embs,prefix2_id,i_token,text_mask):
    img_len = i_token.shape[1]
    prefix2_len = prefix2_id.shape[1]
    final_mask = []
    for i,prefix1 in enumerate(prefixes_embs):
        # print(f"prefix1.shape: {prefix1.shape[1]}")
        len_mask1 = prefix1.shape[0]+img_len+prefix2_len
        # print(f"len_mask1:{len_mask1}")
        mask1 = torch.ones(len_mask1)
        concat_mask = torch.cat([mask1,text_mask[i]],dim=0)
        # print(f"concat_mask: {concat_mask.shape}")
        trunc_mask = concat_mask[:text_mask.shape[1]]
        # print(f"trunc_mask: {trunc_mask.shape}")
        final_mask.append(trunc_mask)
    return torch.stack(final_mask)

def get_merged_retrieved_text_mlm(retrieved_top_k_text,class_lookup_table,ref_lookup_table,b_label):
    ### merge text in a row into a single sentence
    merged_batch_text = []
    merged_b_target = []
    len_retrieved_t = retrieved_top_k_text.shape[1]
    for i,list_t in enumerate(retrieved_top_k_text):
        # merged_batch_text.append(" ".join(list_t))
        list_masked_t = []
        list_target = []
        for j,t in enumerate(list_t):
            labled_t = ""
            if class_lookup_table[ref_lookup_table[t]]==b_label[i]:
                labled_t = "relevant:"
            else:
                labled_t = "irrelevant:"

            sentinel = "<extra_id_" + str(j) + "> "
            masked_t = sentinel + t + "</s>"
            list_masked_t.append(masked_t)
            if j<len_retrieved_t-1:
                labled_t = sentinel + labled_t
                list_target.append(labled_t)
            else:
                labled_t = sentinel + labled_t + " <extra_id_" + str(j+1) + "> "
                list_target.append(labled_t)
            
        # merged_text = "# "+ " # ".join(list_labeled_t)
        merged_text = " ".join(list_masked_t)
        merged_target = " ".join(list_target)
        # merged_batch_text.append(" # ".join(list_t))
        merged_batch_text.append(merged_text)
        merged_b_target.append(merged_target)
    return merged_batch_text,merged_b_target

def count_relevant_text(benign_stat,malignant_stat,retrieved_top_k_text,class_lookup_table,ref_lookup_table,b_label):
    ### merge text in a row into a single sentence
    # merged_batch_text = []
    # b_relevant_text = []
    for i,list_t in enumerate(retrieved_top_k_text):
        # relevant_text = []
        for t in list_t:
            if class_lookup_table[ref_lookup_table[t]]==b_label[i]:
                if b_label[i]==0:#image is bengin
                    if t not in benign_stat:
                        benign_stat[str(t)]=1
                    else:
                        benign_stat[str(t)]+=1
                else:#image is malignant
                    if t not in malignant_stat:
                        malignant_stat[str(t)]=1
                    else:
                        malignant_stat[str(t)]+=1
                # relevant_text.append(t)
        # relevant_text = "# "+ " # ".join([str(t) for t in relevant_text])
        # print(f"relevant_text: {relevant_text}")


def weighted_precision(l_labels,l_precisions):
    # print(f"len l_labels: {len(l_labels)}, len l_precisions: {len(l_precisions)}")
    d = dict()
    for i,l in enumerate(l_labels):
        if l in d.keys():
                d[l].append(l_precisions[i])
        else:
            try:
                d[l] = [l_precisions[i]]
            except:
                print(f"i: {i}")
                sys.exit()
    #class precisions and weights
    precisions = []
    weights = []
    for cls in d.keys():
         ps = d[cls]
         #precision for single class
         cls_precision = sum(ps)/len(ps)#mean precision
         precisions.append(cls_precision)
         #weight of the class
         w = len(ps)/len(l_labels)
         weights.append(w)
    return torch.tensor(precisions)@torch.tensor(weights)

def is_fozen(module):
    for name, param in module.named_parameters():
        print(f"{name}   trainable:  {int(param.requires_grad)}")

def freeze_model(model):
    """Freezes all parameters of the model."""
    for param in model.parameters():
        param.requires_grad = False
        


