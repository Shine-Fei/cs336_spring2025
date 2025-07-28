import os
from typing import BinaryIO, Union

def find_chunk_boundaries(
    file: Union[BinaryIO, str],
    desired_num_chunks: int, 
    split_special_token: Union[bytes, str]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if isinstance(file, str):
        # file is already fully read as string
        assert isinstance(split_special_token, str), "For str input, special token must also be str."
        data = file
        data_len = len(data)
        token_len = len(split_special_token)

        chunk_size = data_len // desired_num_chunks
        chunk_boundaries = [0]

        i = chunk_size
        while len(chunk_boundaries) < desired_num_chunks:
            next_pos = data.find(split_special_token, i)
            if next_pos == -1:
                break
            chunk_boundaries.append(next_pos)
            i = next_pos + token_len

        chunk_boundaries.append(data_len)
        return sorted(set(chunk_boundaries))
    else:

        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))