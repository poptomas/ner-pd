import argparse

from ner_pd.dataset_processors import EmailProcessor
from ner_pd.parsers import NERParser
from ner_pd import ner
from ner_pd.ner import NER
from ner_pd.utils import FormatUtils, benchmark


@benchmark
def prepare_dataset(args: argparse.Namespace) -> EmailProcessor:
    try:
        return EmailProcessor(args)
    except:
        pass


def main(args: argparse.Namespace):
    email_processor_instance = prepare_dataset(args)
    if not email_processor_instance:
        FormatUtils.print_error_with_dataset(args.filename)
        return

    runner = ner.ModelRunner(email_processor_instance, args)
    runner.prepare_directory()

    model_map = NER.get_model_map()
    group_map = NER.get_group_map()

    try:
        if args.mode in group_map.keys():
            for member in model_map.values():
                if member.identifier.startswith(group_map[args.mode]):
                    model = member.model_type(member.identifier)
                    runner.add(model)
        elif args.mode in model_map:
            member = model_map[args.mode]
            model = member.model_type(member.identifier)
            runner.add(model)
        else:
            raise NotImplementedError('Mode "{}" is not supported'.format(args.mode))
    except Exception as exc:
        print(exc)
        return
    runner.process_emails(args.email_count)
    runner.write_stats()


if __name__ == "__main__":
    parser = NERParser()
    args = parser.get_arguments()
    FormatUtils.print_arguments(args)
    main(args)
