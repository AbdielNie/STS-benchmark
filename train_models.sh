# Firstly, fine-tune five pre-trained models., token_type_id shall be modified for some models

# fine-tune bert-base-uncased
python train.py --train_batch_size 8 --valid_batch_size 8 --model_dir ./bert-base-uncased-model --model_name_or_path bert-base-uncased
# fine-tune roberta-base
python train.py --train_batch_size 8 --valid_batch_size 8 --model_dir ./robert-base --model_name_or_path roberta-base
# fine-tune sentence-transformers/stsb-mpnet-base-v2
python train.py --train_batch_size 8 --valid_batch_size 8 --model_dir ./stsb-mpnet-base --model_name_or_path sentence-transformers/stsb-mpnet-base-v2
# fine-tune  cross-encoder/stsb-roberta-base
python train.py --train_batch_size 8 --valid_batch_size 8 --model_dir ./stsb-roberta-base --model_name_or_path cross-encoder/stsb-roberta-base --epochs 0
# fine-tune cross-encoder/stsb-roberta-large
# the batch size is small because the GPU memory can not handle bigger batch size.
python train.py --train_batch_size 1 --valid_batch_size 1 --model_dir ./stsb-roberta-large --model_name_or_path cross-encoder/stsb-roberta-large --epochs 0
# fine-tune microsoft/deberta-base
python train.py --train_batch_size 4 --valid_batch_size 4 --model_dir ./deberta-base --model_name_or_path microsoft/deberta-base
# fine-0tune yoshitomo-matsubara/bert-base-uncased-stsb this is a fine-tuned model
python train.py --train_batch_size 8 --valid_batch_size 8 --model_dir ./bert-base-uncased-stsb --model_name_or_path yoshitomo-matsubara/bert-base-uncased-stsb --epochs 0
# cross-encoder/stsb-TinyBERT-L-4
python train.py --train_batch_size 16 --valid_batch_size 16 --model_dir ./stsb-TinyBERT-L-4 --model_name_or_path cross-encoder/stsb-TinyBERT-L-4 --epochs 0
# cross-encoder/stsb-distilroberta-base
python train.py --train_batch_size 16 --valid_batch_size 16 --model_dir ./stsb-distilroberta-base --model_name_or_path cross-encoder/stsb-distilroberta-base --epochs 0

# Then use the five models to make predictions on test set.

#
python inference.py --model_dir ./bert-base-uncased-model --output_dir ./bert-base-uncased-model-out
python inference.py --model_dir ./bert-base-uncased-model --output_dir ./bert-base-uncased-model-out --dataset validation
#
python inference.py --model_dir ./robert-base --output_dir ./robert-base-out
python inference.py --model_dir ./robert-base --output_dir ./robert-base-out --dataset validation
# #
python inference.py --model_dir ./stsb-mpnet-base --output_dir ./stsb-mpnet-base-out
python inference.py --model_dir ./stsb-mpnet-base --output_dir ./stsb-mpnet-base-out --dataset validation
# #
python inference.py --model_dir ./stsb-roberta-base --output_dir ./stsb-roberta-base-out
python inference.py --model_dir ./stsb-roberta-base --output_dir ./stsb-roberta-base-out --dataset validation
#
python inference.py --model_dir ./stsb-roberta-large --output_dir ./stsb-roberta-large-out
python inference.py --model_dir ./stsb-roberta-large --output_dir ./stsb-roberta-large-out --dataset validation
#
python inference.py --model_dir ./deberta-base --output_dir ./deberta-base-out
python inference.py --model_dir ./deberta-base --output_dir ./deberta-base-out --dataset validation
#
python inference.py --model_dir ./bert-base-uncased-stsb --output_dir ./bert-base-uncased-stsb-out
python inference.py --model_dir ./bert-base-uncased-stsb --output_dir ./bert-base-uncased-stsb-out --dataset validation
# #
python inference.py --model_dir ./stsb-TinyBERT-L-4 --output_dir ./stsb-TinyBERT-L-4-out
python inference.py --model_dir ./stsb-TinyBERT-L-4 --output_dir ./stsb-TinyBERT-L-4-out --dataset validation
#
python inference.py --model_dir ./stsb-distilroberta-base --output_dir ./stsb-distilroberta-base-out
python inference.py --model_dir ./stsb-distilroberta-base --output_dir ./stsb-distilroberta-base-out --dataset validation

# Finally use blending to get better results.
# python blend.py
# look at other files in model_ensemble folder
