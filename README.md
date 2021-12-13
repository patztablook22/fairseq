Branch containing source code for the paper "Sequence Length is a Domain: Length-based Overfitting in Transformer Models":
https://aclanthology.org/2021.emnlp-main.650/

Reproducing the results of the NMT experiments (TODO: expand on these):
0. Download CzEng 2.0

1. Download and preprocess the dataset using custom_examples/translation/prepare-wmt20\* scripts.

2. Preprocess the datasets via Fairseq using wrappers/preprocess_length_domain_translation.sh

3. Run training using wrappers/train_transformer.sh


To cite this paper, you can use the following bibtex entry:
@inproceedings{varis-bojar-2021-sequence,
    title = "Sequence Length is a Domain: Length-based Overfitting in Transformer Models",
    author = "Varis, Dusan  and
      Bojar, Ond{\v{r}}ej",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.650",
    pages = "8246--8257",
    abstract = "Transformer-based sequence-to-sequence architectures, while achieving state-of-the-art results on a large number of NLP tasks, can still suffer from overfitting during training. In practice, this is usually countered either by applying regularization methods (e.g. dropout, L2-regularization) or by providing huge amounts of training data. Additionally, Transformer and other architectures are known to struggle when generating very long sequences. For example, in machine translation, the neural-based systems perform worse on very long sequences when compared to the preceding phrase-based translation approaches (Koehn and Knowles, 2017). We present results which suggest that the issue might also be in the mismatch between the length distributions of the training and validation data combined with the aforementioned tendency of the neural networks to overfit to the training data. We demonstrate on a simple string editing tasks and a machine translation task that the Transformer model performance drops significantly when facing sequences of length diverging from the length distribution in the training data. Additionally, we show that the observed drop in performance is due to the hypothesis length corresponding to the lengths seen by the model during training rather than the length of the input sequence.",
}
