from . import util

def merge_unique_to_single_file(path:str, file_pattern:str, output_file: str) -> None:
    """Generates a single file consisting of unique records from a number of files

    Args:
        path (str): Base path where files are stored
        file_pattern (str): Pattern to match files against
    """

    file_list = util.get_files_startswith(path, file_pattern)
    lines = []
    for file in file_list:
        with open(file) as f:
            for line in f:
                lines.append(line)
    
    lines = set(lines)

    with open(output_file, 'w') as out:
        for line in lines:
            out.write(line)

