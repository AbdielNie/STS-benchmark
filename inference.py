import argparse
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from datasets import load_dataset

from load_data import STSDataLoader
import utils


@torch.no_grad()
def inference(args):
    # Set device
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # # Build data loader
    # sts_dataloader = STSDataLoader(tokenizer, args.max_length)
    # test_data = utils.read_json(os.path.join(args.data_dir, 'sts_test.json'))
    # sts_test_loader = sts_dataloader.get_dataloader(
    #     data=test_data,
    #     batch_size=args.batch_size,
    # )

    # BUILD data loader
    datasets = load_dataset("glue", "stsb")
    sts_dataloader = STSDataLoader(tokenizer)
    test_loader = sts_dataloader.get_dataloader(
        data=list(datasets[args.dataset]),
        batch_size=1,
    )

    print(test_loader)

    # Infer
    utils.make_dirs(args.output_dir)
    output_file = open(os.path.join(args.output_dir, "output_{}.csv".format(args.dataset)), "w")
    for out in test_loader:
        input_ids, attention_mask, token_type_ids, _ = [o.to(device) for o in out]
        output = model(
            input_ids, 
            # token_type_ids=token_type_ids, 
            attention_mask=attention_mask)[0]

        preds = output.detach().cpu().numpy()

        for p in preds:
            score = p[0]
            output_file.write(f"{score}\n")

    output_file.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--dataset", type=str, default="test",
        help='inference on test set or val set'
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_CHANNEL_MODEL", "./model"),
        help='path to load trained model'
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
        help='path to save the output'
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="maximum sequence length",
    )

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
