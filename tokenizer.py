import os
import torch

class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1        
        
        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        
    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class LongTextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab = self.text_to_data(file_path, vocab, extend_vocab, device)
        
    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r') as text:
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab
    
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class ChunkedTextData:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // bsz + 1

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])
        return batches