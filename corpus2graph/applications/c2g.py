"""
generate graph from file

Usage:
    c2g all [--process_num=<process_num> --config=<config> --lang=<lang> --max_window_size=<max_window_size> --min_count=<min_count> --max_vocab_size=<max_vocab_size> --safe_files_number_per_processor=<safe_files_number_per_processor> --file_parser=<file_parser>] <data_dir> <output_dir>
    c2g wordprocessing [--process_num=<process_num> --lang=<lang>] <data_dir> <output_dir>
    c2g sentenceprocessing [--max_window_size=<max_window_size> --process_num=<process_num>] <data_dir> <output_dir>
    c2g wordpairsprocessing [--max_window_size=<max_window_size> --process_num=<process_num> --min_count=<min_count> --max_vocab_size=<max_vocab_size> --safe_files_number_per_processor=<safe_files_number_per_processor>] <data_dir> <output_dir>
    c2g -h | --help
    c2g --version
Options:
    <data_dir>                                                            Set data directory. This script expects
                                                                          all corpus data store in this directory
    <output_dir>                                                          Set output directory. The output graph matrix and
                                                                          other intermediate data will be stored in this directory.
                                                                          see "Output directory" section for more details
    --file_parser=<file_parser>                                           File parser to use supported are json, txt and xml file types.
                                                                          [default: txt]
    --json_attribute=<json_attribute>                                     Key from which to extract the sentences when given a json file
                                                                          [default: maintext]                                                                          
    --lang=<lang>                                                         The language used for stop words, tokenizer, stemmer
                                                                          [default: en]
    --max_window_size=<max_window_size>                                   The maximum window size to generate the word pairs.
                                                                          [default: 5]
    --process_num=<process_num>                                           The number of process you want to use.
                                                                          [default: 3]
    --min_count=<min_count>                                               Mininum count for words.
                                                                          [default: 5]
    --max_vocab_size=<max_vocab_size>                                     The maximum number of words you want to use.
                                                                          [default: 10000]
    --safe_files_number_per_processor=<safe_files_number_per_processor>   Safe files number per processor
                                                                          [default: 200]
    -h --help                                                             Show this screen.
    --version                                                             Show version.
Output directory:
    Three sub-directory will be generated:
    --output_dir
         --dicts_and_encoded_texts                  store the encode data for words
         --edges                                    store the edges data
         --graph                                    store the graph
        ...................................................................
"all" mode:
        Genrerate the graph from corpus.
"wordprocessing" mode:
        Todo
"sentenceprocessing" mode:
        Todo
"wordpairsprocessing" mode:
        Todo
"""

from corpus2graph import Tokenizer, WordProcessing, SentenceProcessing, WordPairsProcessing, util, post_processing
from docopt import docopt
import time
import json

def main():
    arguments = docopt(__doc__, version='1.O.O')

    if arguments['--config']:
        try:
            with open(arguments['--config'], 'r') as f:
                config_file = json.load(f)
            # Merge ClI arguments with those from the config file, the former taking priority
            arguments = dict((str(key), arguments.get(key) or config_file.get(key))
                for key in set(config_file) | set(arguments))
        except FileNotFoundError:
            print(arguments['--config'] + ' does not exist, please check again.') 


    if arguments['all']:
        data_folder = arguments['<data_dir>']
        if not data_folder.endswith('/'):
            data_folder += '/'
        output_folder = arguments['<output_dir>']
        if not output_folder.endswith('/'):
            output_folder += '/'
        dicts_folder = output_folder + 'dicts_and_encoded_texts/'
        edges_folder = output_folder + 'edges/'
        graph_folder = output_folder + 'graph/'

        util.mkdir_p(output_folder)
        util.mkdir_p(dicts_folder)
        util.mkdir_p(edges_folder)
        util.mkdir_p(graph_folder)

        # server setting
        process_num = int(arguments['--process_num'])
        # word processing setting
        lang = str(arguments['--lang'])
        # sentence processing setting
        max_window_size = int(arguments['--max_window_size'])
        # word pairs processing setting
        safe_files_number_per_processor = int(arguments['--safe_files_number_per_processor'])
        max_vocab_size = int(arguments['--max_vocab_size'])
        min_count = int(arguments['--min_count'])
        # file parser settings
        file_parser = arguments['--file_parser']
        json_attribute = arguments['--json_attribute']
        start_time = time.time()
        # wordprocessing
        wp = WordProcessing(output_folder=dicts_folder, language=lang, word_tokenizer='spacy', wtokenizer=Tokenizer.mytok,
                            remove_stop_words=True, remove_numbers=True, replace_digits_to_zeros=True, file_parser=file_parser, json_attribute=json_attribute,
                            remove_punctuations=True, stem_word=False, lowercase=True)
        merged_dict = wp.apply(data_folder=data_folder, process_num=process_num)
        # sentenceprocessing
        sp = SentenceProcessing(dicts_folder=dicts_folder, output_folder=edges_folder,
                                max_window_size=max_window_size, local_dict_extension='.dicloc')
        word_count_all = sp.apply(data_folder=dicts_folder, process_num=process_num)
        # wordpairsprocessing
        wpp = WordPairsProcessing(max_vocab_size=max_vocab_size, min_count=min_count,
                                  dicts_folder=dicts_folder, window_size=max_window_size,
                                  edges_folder=edges_folder, graph_folder=graph_folder,
                                  safe_files_number_per_processor=safe_files_number_per_processor)
        result = wpp.apply(process_num=process_num)
        # wpp.multiprocessing_merge_edges_count_of_a_specific_window_size(process_num=process_num, already_existed_window_size=4)

        post_processing.merge_unique_to_single_file(path=dicts_folder,
            file_pattern='nodes_author',
            output_file=output_folder + '/nodes_author.txt')
        print('time in seconds:', util.count_time(start_time))

    if arguments['wordprocessing']:
        data_folder = arguments['<data_dir>']
        if not data_folder.endswith('/'):
            data_folder += '/'
        output_folder = arguments['<output_dir>']
        if not output_folder.endswith('/'):
            output_folder += '/'
        dicts_folder = output_folder + 'dicts_and_encoded_texts/'
        edges_folder = output_folder + 'edges/'
        graph_folder = output_folder + 'graph/'

        util.mkdir_p(output_folder)
        util.mkdir_p(dicts_folder)
        util.mkdir_p(edges_folder)
        util.mkdir_p(graph_folder)

        # server setting
        process_num = int(arguments['--process_num'])
        # word processing setting
        lang = str(arguments['--lang'])

        start_time = time.time()
        wp = WordProcessing(output_folder=dicts_folder, language=lang, word_tokenizer='spacy', wtokenizer=Tokenizer.mytok,
                            remove_stop_words=True, remove_numbers=True, replace_digits_to_zeros=True,
                            remove_punctuations=True, stem_word=False, lowercase=True)
        merged_dict = wp.apply(data_folder=data_folder, process_num=process_num)
        print('time for word processing in seconds:', util.count_time(start_time))

    if arguments['sentenceprocessing']:
        output_folder = arguments['<output_dir>']
        if not output_folder.endswith('/'):
            output_folder += '/'
        dicts_folder = output_folder + 'dicts_and_encoded_texts/'
        edges_folder = output_folder + 'edges/'
        graph_folder = output_folder + 'graph/'

        util.mkdir_p(edges_folder)
        util.mkdir_p(graph_folder)

        # server setting
        process_num = int(arguments['--process_num'])
        # sentence processing setting
        max_window_size = int(arguments['--max_window_size'])

        start_time = time.time()
        sp = SentenceProcessing(dicts_folder=dicts_folder, output_folder=edges_folder,
                                max_window_size=max_window_size, local_dict_extension='.dicloc')
        word_count_all = sp.apply(data_folder=dicts_folder, process_num=process_num)
        print('time for sentence processing in seconds:', util.count_time(start_time))

    if arguments['wordpairsprocessing']:
        output_folder = arguments['<output_dir>']
        if not output_folder.endswith('/'):
            output_folder += '/'
        dicts_folder = output_folder + 'dicts_and_encoded_texts/'
        edges_folder = output_folder + 'edges/'
        graph_folder = output_folder + 'graph/'

        util.mkdir_p(graph_folder)

        # server setting
        process_num = int(arguments['--process_num'])
        safe_files_number_per_processor = int(arguments['--safe_files_number_per_processor'])
        # sentence processing setting
        max_window_size = int(arguments['--max_window_size'])
        # word pairs processing setting
        max_vocab_size = int(arguments['--max_vocab_size'])
        min_count = int(arguments['--min_count'])

        start_time = time.time()
        wpp = WordPairsProcessing(max_vocab_size=max_vocab_size, min_count=min_count,
                                  dicts_folder=dicts_folder, window_size=max_window_size,
                                  edges_folder=edges_folder, graph_folder=graph_folder,
                                  safe_files_number_per_processor=safe_files_number_per_processor)
        result = wpp.apply(process_num=process_num)
        # wpp.multiprocessing_merge_edges_count_of_a_specific_window_size(process_num=process_num, already_existed_window_size=4)
        print('time for word pairs processing in seconds:', util.count_time(start_time))

if __name__ == '__main__':
    main()