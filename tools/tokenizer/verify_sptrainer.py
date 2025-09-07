import sentencepiece as spm

def verify_tokenizer(model_path: str):
    sp = spm.SentencePieceProcessor(model_file=model_path)

    print(f"✅ Loaded tokenizer with vocab size: {sp.vocab_size()}")

    # ✅ Test samples (CLI, logs, schemas)
    test_sentences = [
        "show ip interface brief",
        "<DATETIME> {ndbmand_R0-0}{1}: [sicon] UUID: <UUID>, ra: <RA> (note): connection established",
        "<TABLE_DEF> tbl_control_process { <EMBEDS> svc_loc; <KEY> svc_loc { type avl; } }",
        "<USER> What is the CPU usage on this router?",
        "interface GigabitEthernet0/0/1",
        "ping 8.8.8.8 count 5",
        "<URL> https://wiki.cisco.com/docs/interface/status"
    ]

    for sentence in test_sentences:
        pieces = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)

        print("\n🔹 Sentence:", sentence)
        print("   ➤ Tokens:", pieces)
        print("   ➤ IDs   :", ids)

if __name__ == "__main__":
    # ✅ Update path to your tokenizer model
    model_path = "/home/selvaraj/training/tokenizer_data/olaf_tokenizer.model"
    verify_tokenizer(model_path)
