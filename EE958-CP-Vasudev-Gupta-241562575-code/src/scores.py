import torch
from sacrebleu import sentence_chrf, sentence_bleu


def translate_batch_for_eval(model, eng_tokenizer, indic_tokenizer, english_texts, device='cuda'):
    """
    Batch translation function for evaluation
    """
    model.eval()
    
    # english tokens
    english_ids = eng_tokenizer.texts_to_sequences(english_texts)
    english_tensor = torch.tensor(english_ids, device=device)

    with torch.no_grad():
        translation_ids = model.generate(english_tensor, max_length=20, temperature=0.0)

    # decode from indic_detokenizer
    translation_array = translation_ids.cpu().numpy()
    indic_texts = indic_tokenizer.sequences_to_texts(translation_array)
    
    return indic_texts


def calculate_translation_metrics(model, 
                                  eng_tokenizer, 
                                  indic_tokenizer, 
                                  en_val_texts, 
                                  hi_val_texts, 
                                  device='cuda', 
                                  num_samples=100):
    """
    Calculate BLEU and chrF scores on validation samples
    """
    # Sample subset for evaluation (for speed)
    if len(en_val_texts) > num_samples:
        # indices = torch.randperm(len(en_val_texts))[:num_samples] # check on random val samples
        indices = torch.arange(start=0, end=num_samples)            # check on same samples
        en_sample = [en_val_texts[i] for i in indices]
        hi_sample = [hi_val_texts[i] for i in indices]
    else:
        en_sample = en_val_texts
        hi_sample = hi_val_texts
    
    # Generate translations
    predictions = translate_batch_for_eval(model, eng_tokenizer, indic_tokenizer, en_sample, device)
    
    # Calculate metrics
    bleu_scores = []
    chrf_scores = []

    for pred, ref in zip(predictions, hi_sample):
        if pred.strip() and ref.strip():
            try:
                # BLEU score
                bleu = sentence_bleu(pred, [ref])
                bleu_scores.append(bleu.score)
                
                # chrF score
                chrf = sentence_chrf(pred, [ref])
                chrf_scores.append(chrf.score)
            except:
                # Skip if there's an error with scoring
                pass

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
    
    return {
        'bleu': avg_bleu,
        'chrf': avg_chrf,
        'num_evaluated': len(bleu_scores)
    }
