import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class SequenceEncoder(nn.Module):

    def __init__(self, encoder_type, **kwargs):
        super(SequenceEncoder, self).__init__()

        if encoder_type == 'lstm':
            self.sequence_encoder = StackedBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'connected_lstm':
            self.sequence_encoder = FullyConnectedBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'residual_lstm':
            self.sequence_encoder = ResidualBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences']
            )
        elif encoder_type == 'lstm+mha':
            self.sequence_encoder = AttentiveFullyConnectedBiLSTM(
                kwargs['lstm_input_size'],
                kwargs['lstm_hidden_size'],
                kwargs['lstm_num_layers'],
                kwargs['lstm_dropout'],
                kwargs['lstm_bidirectional'],
                kwargs['pack_sequences'],
                kwargs['mha_num_layers'],
                kwargs['mha_hidden_size'],
                kwargs['mha_num_heads'],
                kwargs['mha_dropout'],
            )

        self.sequence_state_size = self.sequence_encoder.output_size

    def forward(self, input_sequences, sequence_lengths=None):
        return self.sequence_encoder(input_sequences, sequence_lengths)


class StackedBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout,
        lstm_bidirectional,
        pack_sequences,
    ):
        super().__init__()

        self.pack_sequences = pack_sequences
        self.layer_norm = nn.LayerNorm(lstm_input_size)

        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidirectional,
            batch_first=True)

        self.output_size = lstm_hidden_size if not lstm_bidirectional else 2*lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.layer_norm(input_sequences)

        if self.pack_sequences:
            packed_input = pack_padded_sequence(
                input_sequences,
                sequence_lengths,
                batch_first=True,
                enforce_sorted=False)
        else:
            packed_input = input_sequences

        packed_sequence_encodings, _ = self.lstm(packed_input)

        if self.pack_sequences:
            sequence_encodings, _ = pad_packed_sequence(
                packed_sequence_encodings,
                total_length=total_length,
                batch_first=True)
        else:
            sequence_encodings = packed_sequence_encodings

        return sequence_encodings


class FullyConnectedBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout,
        lstm_bidirectional,
        pack_sequences,
    ):
        super().__init__()

        self.pack_sequences = pack_sequences

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = lstm_input_size
        for _ in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            norm = nn.LayerNorm(_layer_input_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size += 2*lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            self.output_size = lstm_input_size + lstm_num_layers*(2*lstm_hidden_size)
        else:
            self.output_size = lstm_input_size + lstm_num_layers*lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):
        total_length = input_sequences.shape[1]

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            normalized_input_sequences = norm(input_sequences)

            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    normalized_input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
            else:
                packed_input = input_sequences

            packed_sequence_encodings, _ = lstm(packed_input)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = drop(sequence_encodings)
            input_sequences = torch.cat([input_sequences, sequence_encodings], dim=-1)
        
        output_sequences = input_sequences

        return output_sequences


class ResidualBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout,
        lstm_bidirectional,
        pack_sequences,
    ):
        super().__init__()

        self.pack_sequences = pack_sequences
        if lstm_bidirectional:
            self.input_projection = nn.Linear(lstm_input_size, 2*lstm_hidden_size)
            self.input_norm = nn.LayerNorm(2*lstm_hidden_size)
        else:
            self.input_projection = nn.Linear(lstm_input_size, lstm_hidden_size)
            self.input_norm = nn.LayerNorm(lstm_hidden_size)

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = 2*lstm_hidden_size
        for i in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            if lstm_bidirectional:
                norm = nn.LayerNorm(2*lstm_hidden_size)
            else:
                norm = nn.LayerNorm(lstm_hidden_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size = 2*lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            self.output_size = 2*lstm_hidden_size
        else:
            self.output_size = lstm_hidden_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.input_projection(input_sequences)
        input_sequences = self.input_norm(input_sequences)

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
            else:
                packed_input = input_sequences

            packed_sequence_encodings, _ = lstm(packed_input)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = norm(sequence_encodings)
            sequence_encodings = sequence_encodings + input_sequences
            sequence_encodings = drop(sequence_encodings)
            input_sequences = sequence_encodings

        return sequence_encodings


class AttentiveFullyConnectedBiLSTM(nn.Module):

    def __init__(
        self,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout,
        lstm_bidirectional,
        pack_sequences,
        mha_num_layers,
        mha_hidden_size,
        mha_num_heads,
        mha_dropout,
    ):
        super().__init__()

        self.pack_sequences = pack_sequences
        self.input_norm = nn.LayerNorm(lstm_input_size)

        _lstms = []
        _norms = []
        _drops = []
        _layer_input_size = lstm_input_size
        for i in range(lstm_num_layers):
            lstm = nn.LSTM(
                _layer_input_size,
                lstm_hidden_size,
                bidirectional=lstm_bidirectional,
                batch_first=True)
            norm = nn.LayerNorm(2*lstm_hidden_size)
            drop = nn.Dropout(lstm_dropout)

            _layer_input_size = 2*lstm_hidden_size if lstm_bidirectional else lstm_hidden_size
            _lstms.append(lstm)
            _norms.append(norm)
            _drops.append(drop)

        self.lstms = nn.ModuleList(_lstms)
        self.norms = nn.ModuleList(_norms)
        self.drops = nn.ModuleList(_drops)

        if lstm_bidirectional:
            lstm_output_size = lstm_input_size + lstm_num_layers*(2*lstm_hidden_size)
        else:
            lstm_output_size = lstm_input_size + lstm_num_layers*lstm_hidden_size

        _queries = []
        _keys = []
        _values = []
        _mhas = []
        for i in range(mha_num_layers):
            query = nn.Linear(mha_input_size, mha_hidden_size)
            key = nn.Linear(mha_input_size, mha_hidden_size)
            value = nn.Linear(mha_input_size, mha_hidden_size)
            mha = nn.MultiheadAttention(mha_hidden_size, mha_num_heads, dropout=mha_dropout)
            _queries.append(query)
            _keys.append(key)
            _values.append(value)
            _mhas.append(mha)

        self.queries = nn.ModuleList(_queries)
        self.keys = nn.ModuleList(_keys)
        self.values = nn.ModuleList(_values)
        self.mhas = nn.ModuleList(_mhas)

        self.output_size = lstm_output_size

    def forward(self, input_sequences, sequence_lengths=None):

        total_length = input_sequences.shape[1]
        input_sequences = self.input_norm(input_sequences)
        output_sequences = input_sequences

        for lstm, drop, norm in zip(self.lstms, self.drops, self.norms):
            if self.pack_sequences:
                packed_input = pack_padded_sequence(
                    input_sequences,
                    sequence_lengths,
                    batch_first=True,
                    enforce_sorted=False)
            else:
                packed_input = input_sequences

            packed_sequence_encodings, _ = lstm(packed_input)

            if self.pack_sequences:
                sequence_encodings, _ = pad_packed_sequence(
                    packed_sequence_encodings,
                    total_length=total_length,
                    batch_first=True)
            else:
                sequence_encodings = packed_sequence_encodings

            sequence_encodings = drop(sequence_encodings)
            sequence_encodings = norm(sequence_encodings)
            input_sequences = sequence_encodings
            output_sequences = torch.cat([output_sequences, sequence_encodings], dim=-1)

        batch_size = output_sequences.shape[0]
        max_length = output_sequences.shape[1]
        input_mask = torch.arange(max_length).expand(batch_size, max_length).to('cuda' if torch.cuda.is_available() else 'cpu')
        input_mask = input_mask >= sequence_lengths.unsqueeze(1)

        input_sequences = torch.transpose(output_sequences, 0, 1)
        for wq, wk, wv, mha in zip(self.queries, self.keys, self.values, self.mhas):
            q = wq(input_sequences)
            k = wk(input_sequences)
            v = wv(input_sequences)
            output_sequences, _ = mha(q, k, v, key_padding_mask=input_mask)
            input_sequences = output_sequences
        output_sequences = torch.transpose(output_sequences, 0, 1)

        return output_sequences
