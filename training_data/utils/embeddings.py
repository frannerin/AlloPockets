from transformers import EsmTokenizer, EsmModel
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ESM2 model and tokenizer
model_name = 'esm2_t33_650M_UR50D'

path = f"/data/fnerin/huggingface/hub/models--facebook--{model_name}"

with open(f"{path}/refs/main", "r") as f:
    model_dir = f"{path}/snapshots/{f.read().strip()}"
    
tokenizer = EsmTokenizer.from_pretrained(model_dir)
model = EsmModel.from_pretrained(model_dir).to(device)

overlap = 50
max_length = 1024



def get_embs(sequence):
    """
    Return the esm2_t33_650M_UR50D embeddings for the passed sequences.
    
    If the sequence length is higher than the LLM max_length of 1024 - 2 (first and last CLS, EOS tokens), the sequence is split in chunks. Between one chunk and the following, the last 100 and first 100 residues of the respective chunks are shared/overlapping in order to extract embeddings with long-range sequence dependencies, and in the end for the 100 overlapping residues the embeddings of the first 50 are from the first chunk and of the other 50 are from the following chunk.
    """
    # Tokenize the input sequence
    inputs = tokenizer(sequence, return_tensors="pt", padding=False, truncation=False)
    input_ids = inputs['input_ids'].squeeze(0)[1:-1] # without [CLS] and [EOS]
    
    seq_length = len(sequence)
    input_chunks = []
    
    def get_chunks(input_ids, n_chunks, seq_length, max_length, overlap):
        # Split input_ids into chunks with overlap
        chunk_ids = list(range(0, seq_length, int(np.ceil(seq_length/n_chunks))))
    
        input_chunks.append(input_ids[:chunk_ids[1]+overlap])
        
        for i in range(len(chunk_ids)-2):
            input_chunks.append(input_ids[chunk_ids[i+1]-overlap:chunk_ids[i+2]+overlap])
            
        input_chunks.append(input_ids[chunk_ids[-1]-overlap:])
        return input_chunks
    
    
    if seq_length < max_length - 2:
        input_chunks.append(input_ids)
    elif seq_length > max_length - 2:
        n_chunks = np.ceil( seq_length / (max_length-2-overlap) )
        input_chunks = get_chunks(input_ids, n_chunks, seq_length, max_length, overlap)
        while any(len(chunk) > 1022 for chunk in input_chunks):
            n_chunks += 1
            input_chunks = get_chunks(input_ids, n_chunks, seq_length, max_length, overlap)

    assert all(len(i)<max_length-2 for i in input_chunks)
    
    all_embeddings = []
    
    # Process each chunk and extract embeddings
    for i, chunk in enumerate(input_chunks):
        # Add [CLS] and [EOS] tokens
        chunk_with_special_tokens = torch.cat([torch.tensor([tokenizer.cls_token_id]), # , device=device
                                               chunk, 
                                               torch.tensor([tokenizer.eos_token_id])]) # , device=device
        
        # Create the attention mask
        chunk_attention = torch.tensor([1]*len(chunk_with_special_tokens))
        
        # Add padding if necessary
        chunk_with_special_tokens = torch.cat([
            chunk_with_special_tokens,
            torch.tensor(
                [tokenizer.pad_token_id] * (max_length - len(chunk_with_special_tokens)), 
                # device=device
            )
        ])
        # also to the attention mask
        chunk_attention = torch.cat([
            chunk_attention,
            torch.tensor( [0] * (max_length - len(chunk_attention)) )
        ])
        
        no_padding = np.where(chunk_with_special_tokens != 1)
    
        
        
        # Get embeddings
        chunk_inputs = {
            'input_ids': chunk_with_special_tokens.unsqueeze(0).to(device),
            "attention_mask": chunk_attention.unsqueeze(0).to(device)
        }
        with torch.no_grad():
            outputs = model(**chunk_inputs)
        
        # Extract embeddings for the residues (exclude CLS and SEP)
        chunk_embeddings = outputs.last_hidden_state.squeeze(0)[no_padding][1:-1, :] # Exclude padding, CLS (first) and EOS (last)
        
        all_embeddings.append(chunk_embeddings)
    
    # Concatenate embeddings handling overlap
    final_embeddings = []
    
    if len(all_embeddings) == 1:
        final_embeddings = all_embeddings[0].cpu().numpy()
    else:
        final_embeddings.append(all_embeddings[0][:-overlap, :])
        
        for i in range(len(all_embeddings) - 2):
            # Add non-overlapping part from the current chunk
            final_embeddings.append(all_embeddings[i+1][overlap:-overlap, :])
    
        final_embeddings.append(all_embeddings[-1][overlap:, :])
    
        # Concatenate all processed chunks
        final_embeddings = torch.cat(final_embeddings, dim=0).cpu().numpy()

    assert len(final_embeddings) == seq_length
    
    return final_embeddings