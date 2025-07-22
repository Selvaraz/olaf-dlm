import torch
import tiktoken
from ..transformer.OpalGPTModel import OpalGPT
from ..opalmain.opal_main import Opal
import time

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 256, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"transformer_drop_rate": 0.1, # Transformer dropout rate
"attention_drop_rate": 0.1, # Attention dropout rate
"qkv_bias": False # Query-Key-Value bias
}


tokenizer = tiktoken.get_encoding("gpt2")
opalInstance = Opal(
    config=GPT_CONFIG_124M, 
    tokenizer=tokenizer)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_train_test():
    start_time = time.time()
    torch.manual_seed(123)
    model = OpalGPT(GPT_CONFIG_124M)
    model.to(device)
    
    # Create an AdamW optimizer that will update the model's parameters
    # based on the gradients computed during backpropagation.
    # The learning rate is set to 0.0004 and the weight decay is set to 0.1.
    # Weight decay is a regularization technique that adds a penalty term
    # to the loss function for large weights. This helps to prevent overfitting.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10

    # Step1: Load the training data
    # Ths is the total text file that will be used for training.
    # The text file is read and the content is returned as a string.
    total_corpus = opalInstance.loadTrainingData()

    # Step2: Create a DataLoader for the training data and value_loader
    # The DataLoader is  used to load the training data in batches.
    # The batch size is set to 4, the number of workers is set to 4,
    # the maximum length of the sequence is set to the context length,
    # the stride is set to the context length, and the data is shuffled.
    train_ratio = 0.90
    split_idx = int(train_ratio * len(total_corpus))
    train_data = total_corpus[:split_idx]
    val_data = total_corpus[split_idx:]
    total_tokens = len(tokenizer.encode(total_corpus))
    
    training_loader = opalInstance.createOpalDataLoader(
        txt=train_data,
        batch_size=4, 
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False, 
    )

    print(next(iter(training_loader)))

    val_loader = opalInstance.createOpalDataLoader(
        txt=val_data,
        batch_size=4, 
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False, 
    )

    # Sanity check
    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`")
    
    print("Train loader:")

    

    for idx, (x, y) in enumerate(training_loader):
        print(f"Batch {idx}: {x.shape} {y.shape}")

    print("\nValidation loader:")
    for idx, (x, y) in enumerate(val_loader):
        print(f"Batch {idx}: {x.shape} {y.shape}")
    
    train_tokens = 0
    for input_batch, target_batch in training_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)


    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = opalInstance.calc_loss_loader(training_loader, model, device)
        val_loss = opalInstance.calc_loss_loader(val_loader, model, device)
        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)


    train_losses, val_losses, tokens_seen = opalInstance.train_model_simple(model=model, 
                                    train_loader=training_loader,
                                    val_loader=val_loader,
                                    tokenizer=tokenizer,
                                    optimizer=optimizer,
                                    device=device, num_epochs=num_epochs,
                                    eval_iter=5, eval_freq=5,
                                    start_context="Every effort moves you"
                                    )
    epochs_tensor = torch.linspace(1, num_epochs, len(train_losses))
    opalInstance.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)    



def transformer_test():
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(opalInstance.text_to_token_ids(txt1).squeeze(0))
    batch.append(opalInstance.text_to_token_ids(txt2).squeeze(0))
    batch = torch.stack(batch, dim=0)

    model = OpalGPT(GPT_CONFIG_124M)
    out = model(batch)
    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)


    with torch.no_grad():
        logits = model(batch)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    targets = batch[:, 1:]  # Shifted input for targets
    print("inputs", batch)
    print("Targets:\n", targets)
    print(f"Targets batch 1: {opalInstance.token_ids_to_text(targets[0])}")
    print(f"Outputs batch 1:"
    f" {opalInstance.token_ids_to_text(token_ids[0].flatten())}")

    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)


    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Cross-entropy loss:", loss.item())
    print(loss)

    # total_params = sum(p.numel() for p in model.parameters())
    # print("Total number of trainable model parameters: ", total_params)

    # total_size_bytes = total_params * 4
    # total_size_mb = total_size_bytes / (1024 * 1024)
    # print(f"Total size of the model: {total_size_mb:.2f} MB")

    # ## 
    # start_context = "Every effort moves you"
    # model.eval()
    # out = opalInstance.generate_text_simple(
    #     model=model,
    #     idx=opalInstance.text_to_token_ids(start_context),
    #     max_new_tokens=20,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )
    # print("Output:", out)
    # print("Output length:", len(out[0]))


    # decoded_text = opalInstance.token_ids_to_text(out)
    # print(decoded_text)

start_time = time.time()
model_train_test()    
end_time = time.time()
execution_time_in_mins = (end_time - start_time) / 60
print(f"Model training and evaluation completed in {execution_time_in_mins:.2f}) minutes")