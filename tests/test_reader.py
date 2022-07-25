import numpy as np
import os
import pandas as pd
import pytest
import reader


@pytest.fixture
def example_mqreader():
    return reader.MQReader('./tests/testdata/maxquant_results')


@pytest.fixture
def example_fpreader():
    return reader.FPReader('./tests/testdata/fragpipe_results')


def test_that_always_passes():
    assert True


def test_extract_sample_names(example_mqreader):
    protein_table = example_mqreader._read_file('proteins')
    sample_names = reader.extract_sample_names(protein_table, 'Intensity')
    assert len(sample_names) == 18


def test_replace_column_tag():
    df = pd.DataFrame(columns=['Tag', 'Tag A', 'Tag B', 'Something else'])
    old_tag = 'Tag'
    new_tag = 'New'
    new_df = reader._replace_column_tag(df, old_tag, new_tag)
    new_columns = new_df.columns.tolist()
    assert new_columns == ['New', 'New A', 'New B', 'Something else']


def test_rearrange_column_tag():
    df = pd.DataFrame(columns=['Tag1 Text1', 'Tag1 Text2', 'Tag1',
                               'Text1 Tag2', 'Text2 Tag2', 'Tag2'])
    tag = 'Tag1'
    prefixed = False
    new_df = reader._rearrange_column_tag(df, tag, prefixed)
    new_columns = new_df.columns.tolist()
    assert new_columns == ['Text1 Tag1', 'Text2 Tag1', 'Tag1',
                           'Text1 Tag2', 'Text2 Tag2', 'Tag2']

    tag = 'Tag2'
    prefixed = True
    new_df = reader._rearrange_column_tag(df, tag, prefixed)
    new_columns = new_df.columns.tolist()
    assert new_columns == ['Tag1 Text1', 'Tag1 Text2', 'Tag1',
                           'Tag2 Text1', 'Tag2 Text2', 'Tag2']


def test_find_remaining_substrings():
    strings_list = [
        ['Test Sub1', 'Test Sub2', 'Test Sub3', 'Test', 'Test Sub3'],
        ['Sub1 Test', 'Sub2 Test', 'Sub3 Test', 'Test']
    ]
    split_with = 'Test'
    for strings in strings_list:
        substrings = reader._find_remaining_substrings(strings, split_with)
        assert len(substrings) == 3
        assert substrings == ['Sub1', 'Sub2', 'Sub3']


def test_sort_fasta_entries_with_single_entry():
    # Single entry
    fasta_headers = ['x|A|a']
    fastas, proteins, names = reader._sort_fasta_entries(fasta_headers)
    assert fastas == ['x|A|a']
    assert proteins == ['A']
    assert names == ['a']

    fasta_headers = ['x|A|a something else']
    fastas, proteins, names = reader._sort_fasta_entries(fasta_headers)
    assert fastas == ['x|A|a something else']
    assert proteins == ['A']
    assert names == ['a']


def test_sort_fasta_entries_with_multiple_entries():
    # Multiple entries
    fasta_headers = ['x|A|a', 'x|B|b']
    fastas, proteins, names = reader._sort_fasta_entries(fasta_headers)
    assert fastas == ['x|A|a', 'x|B|b']
    assert proteins == ['A', 'B']
    assert names == ['a', 'b']


def test_sort_fasta_entries_with_multiple_unsorted_entries():
    # Multiple unsorted entries
    fasta_headers = ['x|B|b', 'x|A|a']
    fastas, proteins, names = reader._sort_fasta_entries(fasta_headers)
    assert fastas == ['x|A|a', 'x|B|b']
    assert proteins == ['A', 'B']
    assert names == ['a', 'b']


def test_sort_fasta_entries_with_sorting_by_tag():
    # Sorting with sort tags
    fasta_headers = ['x|B|b', 'x|A_end|a_end', 'x|C_A|c_a']
    sorting_tags = {'C_': -1, '_end': 1}
    fastas, proteins, names = reader._sort_fasta_entries(
        fasta_headers, sorting_tags
    )
    assert fastas == ['x|C_A|c_a', 'x|B|b', 'x|A_end|a_end']
    assert proteins == ['C_A', 'B', 'A_end']
    assert names == ['c_a', 'b', 'a_end']


@pytest.mark.parametrize(
    'cols_dropped, cols_remaining',
    [([], ['Col A', 'Col B', 'Col C']),
     (['Col A'], ['Col B', 'Col C']),
     (['Col B', 'Col C'], ['Col A']),
     (['Col B'], ['Col A', 'Col C'])]
)
def test_result_reader_drop_columns(cols_dropped, cols_remaining):
    df = pd.DataFrame(columns=['Col A', 'Col B', 'Col C'])
    base_reader = reader.ResultReader()
    df = base_reader._drop_columns(df, cols_dropped)

    assert set(df.columns) == set(cols_remaining)


@pytest.mark.parametrize(
    'cols_inital, cols_remaining',
    [(['Col A', 'Col B', 'Col C'], ['Col A', 'Col B', 'Col C']),
     (['Col A', 'Drop B', 'Col C'], ['Col A', 'Col C']),
     (['Drop A', 'Col B', 'Drop C'], ['Col B']),
     (['Drop A', 'Drop B', 'Drop C'], [])]
)
def test_result_reader_drop_columns_by_tag(cols_inital, cols_remaining):
    tag = 'Drop'
    df = pd.DataFrame(columns=cols_inital)
    base_reader = reader.ResultReader()
    df = base_reader._drop_columns_by_tag(df, tag)

    assert set(df.columns) == set(cols_remaining)


def test_mqreader_setup(example_mqreader):
    assert os.path.isdir(example_mqreader.data_directory)


def test_mqreader_read_file(example_mqreader):
    # Todo: change test to parental class ResultReader
    protein_table = example_mqreader._read_file('proteins')
    assert isinstance(protein_table, pd.DataFrame)
    assert protein_table.shape == (7, 200)


def test_mqreader_rename_columns(example_mqreader):
    # Todo: change test to parental class ResultReader
    protein_table = example_mqreader._read_file('proteins')
    assert sum(protein_table.columns.str.count('MS/MS count')) > 0
    assert sum(protein_table.columns.str.count('Spectral count')) == 0
    assert ('Peptides' in protein_table.columns)

    old_column_name = 'Peptides'
    new_column_name = 'Total peptides'
    example_mqreader.column_mapping = {old_column_name: new_column_name}
    old_column_tag = 'MS/MS count'
    new_column_tag = 'Spectral count'
    example_mqreader.column_tag_mapping = {old_column_tag: new_column_tag}
    prefixed = False

    protein_table = example_mqreader._rename_columns(protein_table,
                                                     prefixed)

    assert ('Peptides' not in protein_table.columns)
    assert ('Total peptides' in protein_table.columns)

    assert sum(protein_table.columns.str.count('MS/MS count')) == 0
    assert sum(protein_table.columns.str.count('Spectral count')) == 19

    columns_ending_with_new_tag = [
        c.endswith(new_column_tag) for c in protein_table.columns]
    assert sum(columns_ending_with_new_tag) == 19


def test_mqreader_drop_decoy(example_mqreader):
    protein_table = example_mqreader._read_file('proteins')
    protein_table = example_mqreader._drop_decoy(protein_table)
    is_decoy = protein_table['Majority protein IDs'].str.contains('REV__')
    assert not is_decoy.any()


def test_mqreader_drop_idbysite(example_mqreader):
    protein_table = example_mqreader._read_file('proteins')
    protein_table = example_mqreader._drop_idbysite(protein_table)
    is_idbysite = (protein_table['Only identified by site'] == '+')
    assert not is_idbysite.any()


def test_mqreader_rearrange_proteins(example_mqreader):
    df = pd.DataFrame({
        'Majority protein IDs': ['B;A;C', 'D', 'E;F', 'G;H;I'],
        'Peptide counts (all)': ['5;5;3', '3', '6;3', '6;6;6'],
    })
    leading_proteins = ['A;B', 'D', 'E', 'G;H;I']
    representative_protein = ['A', 'D', 'E', 'G']
    protein_reported_by_software = ['B', 'D', 'E', 'G']

    df = example_mqreader._rearrange_proteins(df)
    assert df['Leading proteins'].tolist() == leading_proteins
    assert df['Representative protein'].tolist() == representative_protein
    assert df['Protein reported by software'].tolist() == protein_reported_by_software


def test_fpreader_setup(example_fpreader):
    assert os.path.isdir(example_fpreader.data_directory)


def test_fpreader_rearrange_proteins(example_fpreader):
    df = pd.DataFrame({
        'Protein': [
            'x|B|b', 'x|D|d', 'x|E|e', 'x|G|g'],
        'Indistinguishable Proteins': [
            'x|A|a', '', '', 'x|H|h, x|I|i'],
    })
    leading_proteins = ['A;B', 'D', 'E', 'G;H;I']
    representative_protein = ['A', 'D', 'E', 'G']
    protein_reported_by_software = ['B', 'D', 'E', 'G']

    df = example_fpreader._rearrange_proteins(df)
    assert df['Leading proteins'].tolist() == leading_proteins
    assert df['Representative protein'].tolist() == representative_protein
    assert df['Protein reported by software'].tolist() == protein_reported_by_software


class TestAddIbaqIntensities:
    @pytest.fixture(autouse=True)
    def _init_qtable(self,):
        self.table = pd.DataFrame({
            'peptides': [2, 2],
            'ibaq_petides': [2, 2],
            'intensity': [100.0, 200.0],
        })

    def test_ibaq_intensity_added(self):
        reader.add_ibaq_intensities(
            self.table,
            peptide_column='peptides',
            ibaq_peptide_column='ibaq_petides',
            intensity_tag='intensity',
            ibaq_tag='ibaq',
        )
        assert 'ibaq' in self.table.columns

    @pytest.mark.parametrize(
        'compare_ibaq_intensities, num_petides, normalize_intensity',
        [(np.equal, 2, False), (np.less, 1, False), (np.greater, 3, False),
         (np.equal, 2, True), (np.equal, 1, True), (np.equal, 3, True)]
    )
    def test_correct_ibaq_intensities(self, compare_ibaq_intensities, num_petides, normalize_intensity):
        self.table['peptides'] = num_petides
        reader.add_ibaq_intensities(
            self.table, normalize_total_intensity=normalize_intensity,
            peptide_column='peptides', ibaq_peptide_column='ibaq_petides',
            intensity_tag='intensity', ibaq_tag='ibaq',
        )
        assert np.all(compare_ibaq_intensities(self.table['ibaq'], self.table['intensity']))
