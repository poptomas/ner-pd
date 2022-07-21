import argparse

from ner_pd.process_sentences import AnnotationRunner
from ner_pd.parsers import AnnotationParser
from ner_pd.utils import FilesystemUtils, FormatUtils
from ner_pd.ner import NER


def main(args: argparse.Namespace):
    FilesystemUtils.clean_previous_run(args.outdir)
    model_map = NER.get_model_map()
    group_map = NER.get_group_map()
    runner = AnnotationRunner(args)
    try:
        runner.prepare()
    except:
        FormatUtils.print_error_with_dataset(args.filename)
        return
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
    runner.run()
    runner.evaluate_findings()
    FilesystemUtils.clean_up_besides("csv", args.outdir)


if __name__ == "__main__":
    parser = AnnotationParser()
    args = parser.get_arguments()
    FormatUtils.print_arguments(args)
    main(args)
