import logging
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import f1_score,accuracy_score
from transformers import (
    CLIPModel, CLIPProcessor, CLIPTokenizerFast,
    T5Tokenizer, T5ForConditionalGeneration
)
from util import (
    get_retrieved_top_k_text, id_to_emb, generate_projected_img_emb, kaiming_init_weights,
    get_merged_retrieved_and_relevant_text, get_text_ids, text_to_emb, get_label_ids,
    construct_multimodal_sentence_emb, generate_sentence_mask, get_randomized_retrieved_text_add_noise,
    get_class_lookup_table, get_chosen_knowldge_class, weighted_precision
)
from knowledge_base import Knowledge_base
from modules import VL_model
from custom_dataset import BreaKHis

# ========== Utility Functions ==========
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_text_features(text,plip_tokenizer,plip_model):
    inputs = plip_tokenizer(text, padding='max_length',
                        max_length=77, truncation=True, return_tensors="pt").to(device)
    text_features = plip_model.get_text_features(**inputs)
    return text_features

def clean_strings(strings):
    """
    Cleans a list of strings by removing <pad> tokens and extracting text after "#" symbols.
    Returns a list of cleaned strings, each containing only the text after "#" symbols.
    """
    cleaned_strings = []
    for string in strings:
        cleaned_string = string.replace("<pad>", "").replace("</s>", "")
        extracted_text = [text.strip().replace(".", "") for text in cleaned_string.split("#") if text.strip()]
        cleaned_strings.append(extracted_text)
    return cleaned_strings

def appearance_check(batch_output_text, b_labels, benign_kn_base, malg_kn_base):
    """
    Checks the appearance of knowledge base items in the batch output text.
    Returns precision, recall, and predictions for each sample.
    """
    b_kn_base = [t.replace(".", "") for t in benign_kn_base]
    m_kn_base = [t.replace(".", "") for t in malg_kn_base]
    b_kn_base_set = set(b_kn_base)
    m_kn_base_set = set(m_kn_base)
    b_preds, l_precisions, l_recalls = [], [], []
    num_over_threshold_per_b = 0

    for i, output_text in enumerate(batch_output_text):
        num_text = len(output_text)
        num_rel = num_text
        single_img_label = b_labels[i]
        for item in output_text:
            if (single_img_label == 0 and item not in b_kn_base_set) or \
               (single_img_label == 1 and item not in m_kn_base_set):
                num_rel -= 1
        precision = num_rel / num_text if num_text > 0 else 0
        recall = num_rel / 90 if num_text > 0 else 0  # 90 is a hardcoded class size
        l_precisions.append(precision)
        l_recalls.append(recall)
        p = single_img_label if precision > 0.5 else 1 - single_img_label
        b_preds.append(p)
        if precision > 0.5:
            num_over_threshold_per_b += 1
    return l_precisions, l_recalls, num_over_threshold_per_b, b_preds

def get_data_loaders(dataset, batch_size):
    """Split dataset and return DataLoaders for train, val, and test."""
    all_indices = torch.arange(len(dataset))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.05, random_state=seed)
    _, test_indices = train_test_split(test_indices, test_size=0.5, random_state=seed)
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
# ========== Model and Tokenizer Loading ==========
def load_models_and_tokenizers():
    """Load all models and tokenizers required for training."""
    plip_model = CLIPModel.from_pretrained(plip_model_path).to(device)
    plip_model.eval()
    image_encoder = plip_model.vision_model
    image_encoder.eval()
    plip_processor = CLIPProcessor.from_pretrained(plip_processor_path)
    plip_tokenizer = CLIPTokenizerFast.from_pretrained(plip_processor_path)
    for param in plip_model.parameters():
        param.requires_grad = False

    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(device)
    t5_model.eval()
    for param in t5_model.parameters():
        param.requires_grad = False
    return plip_model, image_encoder, plip_processor, plip_tokenizer, t5_tokenizer, t5_model
# ========== Knowledge Base Preparation ==========
def prepare_knowledge_base(llm_kn):
    """Prepare knowledge base and lookup tables."""
    knowledge_base, benign_knowledge_base, malignant_knowledge_base = [], [], []
    know_b = Knowledge_base()
    class_lookup_table = get_class_lookup_table()
    ref_lookup_table = dict()
    chosen_knowldge_class = get_chosen_knowldge_class(llm_kn)
    for sub_base_name, sub_base_v in know_b.knowledge_base_dict.items():
        if sub_base_name not in chosen_knowldge_class:
            logger.info(f"Unchosen: {sub_base_name}")
            continue
        knowledge_base.extend(sub_base_v)
        for t in sub_base_v:
            ref_lookup_table[t] = sub_base_name
            if class_lookup_table[sub_base_name] == 0:
                benign_knowledge_base.append(t)
            if class_lookup_table[sub_base_name] == 1:
                malignant_knowledge_base.append(t)
    logger.info(f"Knowledge base size: {len(knowledge_base)}")
    return knowledge_base, benign_knowledge_base, malignant_knowledge_base, class_lookup_table, ref_lookup_table

def zero_shot_epoch_eval(trained_vlmodel,transform, plip_processor, plip_model, image_encoder, 
    t5_tokenizer, t5_model, knowledge_base, emb_knowledge_base, benign_knowledge_base, malignant_knowledge_base,
                        class_lookup_table, ref_lookup_table, prefix1_emb, prefix2_emb, prefix1_id, prefix2_id):
    ## Initialize zero-shot BreaKHis test dataset
    zs_test_set = BreaKHis(
        data_file_path='../datasets/breaKHis_zero-shot.json',
        transform=transform
    )
    all_indices = torch.arange(len(zs_test_set))
    zs_test_dataset = torch.utils.data.Subset(zs_test_set, all_indices)
    zs_test_loader = DataLoader(zs_test_dataset, batch_size=zs_b_size, shuffle=False)  # No shuffling
    trained_vlmodel.eval()
    image_encoder.eval()
    total_orig_precisions = []
    total_vlm_precisions = []
    predictions = []
    labels_metr = []
    with torch.no_grad():
        for i, (b_img, b_label) in enumerate(zs_test_loader):
            if -1 in b_label:
                logger.warning(f"Skipping batch {i} due to error data.")
                continue
            k = random.randint(5, 15)
            labels_metr.extend(b_label)
            b_size = b_img.shape[0]
            b_img = b_img.to(device)
            # get vlm projected image embeddings
            proj_img_emb = generate_projected_img_emb(trained_vlmodel, image_encoder, plip_processor, b_img, device)
            # get plip's original image embeddings
            b_img_pixels = plip_processor(images=b_img, return_tensors="pt", padding=True, do_rescale=False).pixel_values
            b_img_pixels = b_img_pixels.to(device)
            b_img_emb = plip_model.get_image_features(b_img_pixels)
            # Retrieve top-k text
            retrieved_top_k_text = get_retrieved_top_k_text(knowledge_base, emb_knowledge_base, b_img_emb, k=k)
            merged_batch_text, b_relevant_text = get_merged_retrieved_and_relevant_text(
                retrieved_top_k_text, class_lookup_table, ref_lookup_table, b_label
            )
            # Clean retrieved text
            cleaned_retri_t = clean_strings(merged_batch_text)
            b_retri_precisions, _, _, _ = appearance_check(
                cleaned_retri_t, b_label, benign_knowledge_base, malignant_knowledge_base
            )
            total_orig_precisions.extend(b_retri_precisions)
            # Process multimodal embeddings
            b_retrieved_text_emb = text_to_emb(
                merged_batch_text, t5_tokenizer, pad_config="max_length", max_length=max_source_length, model=t5_model, device=device
            )
            multimodal_sentence_emb = construct_multimodal_sentence_emb(
                prefix1_emb, proj_img_emb, prefix2_emb, b_retrieved_text_emb, max_source_length
            ).to(device)
            # Generate outputs
            ref_masks = t5_tokenizer(
                merged_batch_text,
                padding="max_length",
                max_length=max_source_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            ).attention_mask
            mask = generate_sentence_mask(b_size, prefix1_id, prefix2_id, proj_img_emb, ref_masks).to(device)
            output_ids = t5_model.generate(inputs_embeds=multimodal_sentence_emb, attention_mask=mask, max_length=150)
            output = t5_tokenizer.batch_decode(output_ids)
            cleaned_output = clean_strings(output)
            # Check appearance and calculate metrics
            b_vlm_precisions, _, _, b_preds = appearance_check(
                cleaned_output, b_label, benign_knowledge_base, malignant_knowledge_base
            )
            predictions.extend(b_preds)
            total_vlm_precisions.extend(b_vlm_precisions)
            # Calculate metrics
            orig_median_precision = np.mean(np.array(total_orig_precisions))
            vlm_median_precision = np.mean(np.array(total_vlm_precisions))
            vlm_f1 = f1_score(labels_metr, predictions)
            vlm_w_f1 = f1_score(labels_metr, predictions, average='weighted')
            vlm_acc = accuracy_score(labels_metr, predictions)
            print(f"orig precision: {orig_median_precision}")
            print(f"vlm precision: {vlm_median_precision}")
            print(f"vlm_f1: {vlm_f1}")
            print(f"vlm_w_f1: {vlm_w_f1}")
            print(f"cls_acc_vlm: {vlm_acc}")
    # Calculate metrics
    orig_median_precision = np.mean(np.array(total_orig_precisions))
    vlm_median_precision = np.mean(np.array(total_vlm_precisions))
    vlm_f1 = f1_score(labels_metr, predictions)
    vlm_w_f1 = f1_score(labels_metr, predictions, average='weighted')
    vlm_acc = accuracy_score(labels_metr, predictions)
    print(f"orig precision: {orig_median_precision}")
    print(f"vlm precision: {vlm_median_precision}")
    print(f"vlm_f1: {vlm_f1}")
    print(f"vlm_w_f1: {vlm_w_f1}")
    print(f"cls_acc_vlm: {vlm_acc}")
    wandb.log({"vlm_f1": vlm_f1,"cls_acc_vlm": vlm_acc})
    return vlm_f1,vlm_w_f1,vlm_acc

# ========== Training and Evaluation ==========
def train_for_one_epoch(retri_random, vlmodel, optimizer, pbar, dataset_loader, image_processor, t5_model,
     t5_tokenizer, plip_model, image_encoder, knowledge_base, emb_knowledge_base, benign_knowledge_base,
     malignant_knowledge_base, class_lookup_table, ref_lookup_table, prefix1_emb, prefix2_emb, prefix1_id, prefix2_id):
    """Train for one epoch."""
    total_epoch_loss = 0
    for i, (b_img, b_label) in enumerate(dataset_loader):
        if -1 in b_label:
            logger.warning(f"Skipping batch {i} due to error data.")
            continue
        k = random.randint(5, 15)
        b_size = b_img.shape[0]
        b_img = b_img.to(device)
        #get projected image embeddings of the batch
        proj_img_emb = generate_projected_img_emb(vlmodel, image_encoder, image_processor, b_img, device)
        b_img_pixels = image_processor(images=b_img, return_tensors="pt", padding=True, do_rescale=False).pixel_values.to(device)
        b_img_emb = plip_model.get_image_features(b_img_pixels)
        #get top-k retrieved text
        retrieved_top_k_text = get_retrieved_top_k_text(knowledge_base, emb_knowledge_base, b_img_emb, k=k)
        merged_batch_text, b_relevant_text = get_randomized_retrieved_text_add_noise(
            random_noise_rate, retri_random, benign_knowledge_base, malignant_knowledge_base,
            retrieved_top_k_text, class_lookup_table, ref_lookup_table, b_label
        )
        b_retrieved_text_emb = text_to_emb(merged_batch_text, t5_tokenizer, pad_config="max_length",
                                           max_length=max_source_length, model=t5_model, device=device)
        #visual+textual token embs
        multimodal_sentence_emb = construct_multimodal_sentence_emb(
            prefix1_emb, proj_img_emb, prefix2_emb, b_retrieved_text_emb, max_source_length
        ).to(device)
        ref_masks = t5_tokenizer(
            merged_batch_text, padding="max_length", max_length=max_source_length,
            truncation=True, return_tensors="pt", add_special_tokens=False
        ).attention_mask
        mask = generate_sentence_mask(b_size, prefix1_id, prefix2_id, proj_img_emb, ref_masks).to(device)
        #relevant text ids of the batch of images
        label_ids = get_label_ids(b_relevant_text, t5_tokenizer, max_target_length).to(device)
        loss = t5_model(inputs_embeds=multimodal_sentence_emb, attention_mask=mask, labels=label_ids).loss
        if vlmodel.training:
            wandb.log({"training batch loss": loss.item()})
        else:
            wandb.log({"testing batch loss": loss.item()})
        total_epoch_loss += loss.item()
        if optimizer is not None:
            loss = loss / accum_iter
            loss.backward(retain_graph=True)
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(dataset_loader)):
                optimizer.step()
                optimizer.zero_grad()
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(value=loss.item())
        torch.cuda.empty_cache()
    epoch_loss = total_epoch_loss / len(dataset_loader)
    return epoch_loss

def train(epochs, vlmodel, image_processor, learning_rate, train_loader, t5_model, t5_tokenizer, plip_model,
           image_encoder, knowledge_base, emb_knowledge_base, benign_knowledge_base, malignant_knowledge_base,
             class_lookup_table, ref_lookup_table, prefix1_emb, prefix2_emb, prefix1_id, prefix2_id,transform):
    """Main training loop."""
    optimizer = torch.optim.AdamW([
        {'params': vlmodel.parameters(), 'lr': learning_rate, 'weight_decay': 1e-06},
    ], betas=(0.9, 0.98), eps=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    total_iter = len(train_loader)
    start_time = time.time()
    max_score = 0
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch + 1}")
        vlmodel.train()
        with tqdm(total=total_iter, desc="Processing") as pbar:
            train_epoch_loss = train_for_one_epoch(
                retri_random, vlmodel, optimizer, pbar, train_loader, image_processor,
                t5_model, t5_tokenizer, plip_model, image_encoder, knowledge_base, emb_knowledge_base,
                benign_knowledge_base, malignant_knowledge_base, class_lookup_table, ref_lookup_table,
                prefix1_emb, prefix2_emb, prefix1_id, prefix2_id
            )
            # Validation after each epoch
            vlmodel.eval()
            vlm_f1,vlm_w_f1,cls_acc_vlm = zero_shot_epoch_eval(vlmodel,transform, image_processor, 
            plip_model, image_encoder, t5_tokenizer, t5_model, knowledge_base, emb_knowledge_base, benign_knowledge_base, 
            malignant_knowledge_base, class_lookup_table, ref_lookup_table, prefix1_emb, prefix2_emb, prefix1_id, prefix2_id)
            #saving the best check point
            score = cls_acc_vlm+vlm_f1
            if score > max_score:
                max_score = cls_acc_vlm+vlm_f1
                torch.save(vlmodel, f'{model_name}.pth')
            wandb.log({"epoch": epoch+1})
            print(f"epoch: {epoch}, vlm_f1: {vlm_f1}, vlm_w_f1: {vlm_w_f1}, cls_acc_vlm: {cls_acc_vlm}")
            scheduler.step()
            logger.info(f"Train loss: {train_epoch_loss}")
            wandb.log({"train epoch loss": train_epoch_loss})
    end_time = time.time()
    duration = (end_time - start_time) / 60
    logger.info(f"Training completed in {duration:.2f} mins")

# ========== Configuration ==========
device = torch.device("cuda:1")
seed = 42
epochs = 12
learning_rate = 0.05
batch_size = 5
accum_iter = 3
dropout = 0.39
max_source_length = 316
max_target_length = max_source_length
retri_random = True
random_noise_rate = 1
zs_b_size = 50  # Zero-shot batch size
model_name = 't5_large_gpt4_kn3'
project_name = "vlm project"
# data_file_path='../datasets/breaKHis_train_zero-shot1000.json'
data_file_path = '../datasets/breaKHis_train_zero-shot19456.json'
plip_model_path = "../models/plip_model"
plip_processor_path = "../models/plip_processor"
t5_model_path = "../models/flan-t5_large"
llm_kn = "gpt"
# ========== Logging ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ========== Main Script ==========
def main():
    set_seed(seed)
    plip_model, image_encoder, plip_processor, plip_tokenizer, t5_tokenizer, t5_model = load_models_and_tokenizers()
    knowledge_base, benign_knowledge_base, malignant_knowledge_base, class_lookup_table, ref_lookup_table = prepare_knowledge_base(llm_kn)
    emb_knowledge_base = get_text_features(knowledge_base,plip_tokenizer,plip_model)
    knowledge_base = np.array(knowledge_base)
    # logger.info(f"Benign examples: {benign_knowledge_base[:3]}")
    # logger.info(f"Malignant examples: {malignant_knowledge_base[:3]}")
    # Prefixes
    prefix1 = "Image: "
    prefix2 = ", text: "
    prefix1_id = get_text_ids(prefix1, t5_tokenizer, pad_config="longest", max_length=max_source_length)
    prefix2_id = get_text_ids(prefix2, t5_tokenizer, pad_config="longest", max_length=max_source_length)
    prefix1_emb = id_to_emb(prefix1_id, t5_model, device)
    prefix2_emb = id_to_emb(prefix2_id, t5_model, device)
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = BreaKHis(data_file_path=data_file_path, transform=transform)
    train_loader = get_data_loaders(dataset, batch_size)
    # wandb setup
    run = wandb.init(
        # mode="disabled",
        project=project_name,
        config={
            "t5_model_path": t5_model_path,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "dropout": dropout,
            "batch_size": batch_size,
            "accum_iter": accum_iter,
            "random_noise_rate": random_noise_rate,
            "retri_random": retri_random,
            "model_name": model_name,
            "llm_kn": llm_kn,
        },
    )
    # Model initialization, trainig the image-language projector
    img_emb_dim = image_encoder.config.hidden_size
    lang_dim = t5_model.config.d_model
    vlmodel = VL_model(img_emb_dim, lang_dim, dropout=dropout)
    vlmodel.apply(kaiming_init_weights)
    # vlmodel = torch.load('../code/lora_vlmodel30.pth',weights_only=False)#30 epochs
    vlmodel = vlmodel.to(device)
    vlmodel.train()
    # Training
    train(
        epochs, vlmodel, plip_processor, learning_rate, train_loader,
        t5_model, t5_tokenizer, plip_model, image_encoder, knowledge_base, emb_knowledge_base,
        benign_knowledge_base, malignant_knowledge_base, class_lookup_table, ref_lookup_table,
        prefix1_emb, prefix2_emb, prefix1_id, prefix2_id,transform
    )

if __name__ == "__main__":
    main()