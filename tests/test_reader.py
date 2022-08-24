import os

import numpy as np
import pandas as pd
import pytest

import msreport.reader


@pytest.fixture
def example_fpreader():
    return msreport.reader.FPReader('./tests/testdata/fragpipe')


def test_that_always_passes():
    assert True


def test_extract_sample_names():
    reader = msreport.reader.ResultReader()
    reader._add_data_directory('./tests/testdata/common')
    table = reader._read_file('table.txt')
    sample_names = msreport.reader.extract_sample_names(table, 'A_tag')
    assert set(sample_names) == set(['Column 1', 'Column 2', 'Column 3'])


def test_replace_column_tag():
    df = pd.DataFrame(columns=['Tag', 'Tag A', 'Tag B', 'Something else'])
    old_tag = 'Tag'
    new_tag = 'New'
    new_df = msreport.reader._replace_column_tag(df, old_tag, new_tag)
    new_columns = new_df.columns.tolist()
    assert new_columns == ['New', 'New A', 'New B', 'Something else']


def test_rearrange_column_tag():
    df = pd.DataFrame(columns=['Tag1 Text1', 'Tag1 Text2', 'Tag1',
                               'Text1 Tag2', 'Text2 Tag2', 'Tag2'])
    tag = 'Tag1'
    prefixed = False
    new_df = msreport.reader._rearrange_column_tag(df, tag, prefixed)
    new_columns = new_df.columns.tolist()
    assert new_columns == ['Text1 Tag1', 'Text2 Tag1', 'Tag1',
                           'Text1 Tag2', 'Text2 Tag2', 'Tag2']

    tag = 'Tag2'
    prefixed = True
    new_df = msreport.reader._rearrange_column_tag(df, tag, prefixed)
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
        substrings = msreport.reader._find_remaining_substrings(strings, split_with)
        assert len(substrings) == 3
        assert substrings == ['Sub1', 'Sub2', 'Sub3']


@pytest.mark.parametrize(
    'input, expected_fastas, expected_proteins, expected_names',
    [(['x|A|a'], ['x|A|a'], ['A'], ['a']),
     (['x|A|a something else'], ['x|A|a something else'], ['A'], ['a']),
     (['x|A|a', 'x|B|b'], ['x|A|a', 'x|B|b'], ['A', 'B'], ['a', 'b']),
     (['x|B|b', 'x|A|a'], ['x|A|a', 'x|B|b'], ['A', 'B'], ['a', 'b']),
     ]
)
def test_sort_fasta_entries_(input, expected_fastas, expected_proteins, expected_names):
    fastas, proteins, names = msreport.reader._sort_fasta_entries(input)
    assert fastas == expected_fastas
    assert proteins == expected_proteins
    assert names == expected_names


def test_sort_fasta_entries_with_sorting_by_tag():
    fasta_headers = ['x|B|b', 'x|Apost|a_post', 'x|preA|pre_a']
    sorting_tags = {'pre': -1, 'post': 1}
    fastas, proteins, names = msreport.reader._sort_fasta_entries(
        fasta_headers, sorting_tags
    )
    assert fastas == ['x|preA|pre_a', 'x|B|b', 'x|Apost|a_post']
    assert proteins == ['preA', 'B', 'Apost']
    assert names == ['pre_a', 'b', 'a_post']


def test_sort_proteins_by_tag():
    proteins = ['A', 'Apost', 'preA']
    sorting_tags = {'pre': -1, 'post': 1}
    sorted_proteins = msreport.reader._sort_proteins_by_tag(proteins, sorting_tags)
    assert sorted_proteins == ['preA', 'A', 'Apost']


class TestSortLeadingProteins:
    def test_without_args(self):
        df = pd.DataFrame({
            'Leading proteins': ['B;A', 'D', 'E;F', 'G;I;H'],
        })
        leading_proteins = ['A;B', 'D', 'E;F', 'G;H;I']
        representative_protein = ['A', 'D', 'E', 'G']

        df = msreport.reader._sort_leading_proteins(df)
        assert df['Leading proteins'].tolist() == leading_proteins
        assert df['Representative protein'].tolist() == representative_protein

    def test_with_contamination_tag(self):
        df = pd.DataFrame({
            'Leading proteins': ['B;A', 'D', 'E;F', 'Gtag;I;H'],
        })
        leading_proteins = ['A;B', 'D', 'E;F', 'H;I;Gtag']
        representative_protein = ['A', 'D', 'E', 'H']

        df = msreport.reader._sort_leading_proteins(df, contaminant_tag='tag')
        assert df['Leading proteins'].tolist() == leading_proteins
        assert df['Representative protein'].tolist() == representative_protein

    def test_with_special_proteins(self):
        df = pd.DataFrame({
            'Leading proteins': ['B;A', 'D', 'E;F', 'G;I;H'],
        })
        leading_proteins = ['A;B', 'D', 'F;E', 'H;I;G']
        representative_protein = ['A', 'D', 'F', 'H']

        df = msreport.reader._sort_leading_proteins(df, special_proteins=['F', 'I', 'H'])
        assert df['Leading proteins'].tolist() == leading_proteins
        assert df['Representative protein'].tolist() == representative_protein


class TestAddIbaqIntensities:
    @pytest.fixture(autouse=True)
    def _init_qtable(self,):
        self.table = pd.DataFrame({
            'peptides': [2, 2],
            'ibaq_petides': [2, 2],
            'intensity': [100.0, 200.0],
        })

    def test_ibaq_intensity_added(self):
        msreport.reader.add_ibaq_intensities(
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
        msreport.reader.add_ibaq_intensities(
            self.table, normalize_total_intensity=normalize_intensity,
            peptide_column='peptides', ibaq_peptide_column='ibaq_petides',
            intensity_tag='intensity', ibaq_tag='ibaq',
        )
        assert np.all(compare_ibaq_intensities(self.table['ibaq'], self.table['intensity']))


class TestResultReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.ResultReader()
        self.reader._add_data_directory('./tests/testdata/common')
        self.reader.filenames = {'table': 'table.txt'}
        self.table_nrows = 5
        self.table_ncolumns = 8

    def test_read_file_with_filename_lookup(self):
        table = self.reader._read_file('table')
        assert isinstance(table, pd.DataFrame)
        assert table.shape == (self.table_nrows, self.table_ncolumns)

    def test_read_file_with_filename(self):
        table = self.reader._read_file('table.txt')
        assert isinstance(table, pd.DataFrame)
        assert table.shape == (self.table_nrows, self.table_ncolumns)

    def test_rename_columns_with_mapping(self):
        table = self.reader._read_file('table.txt')
        self.reader.column_mapping = {'Column 1': 'Renamed column'}
        self.reader.column_tag_mapping = {}
        self.reader.sample_column_tags = []

        assert 'Column 1' in table.columns
        assert 'Renamed column' not in table.columns
        table = self.reader._rename_columns(table, prefix_tag=False)
        assert 'Column 1' not in table.columns
        assert 'Renamed column' in table.columns

    def test_rename_columns_with_column_tag_mapping(self):
        table = self.reader._read_file('table.txt')
        self.reader.column_mapping = {}
        self.reader.column_tag_mapping = {'Another_tag': 'B_tag'}
        self.reader.sample_column_tags = []
        assert 'B_tag Column 1' not in table.columns
        table = self.reader._rename_columns(table, prefix_tag=False)
        assert all(['Another_tag' not in c for c in table.columns])
        assert 'B_tag Column 1' in table.columns

    @pytest.mark.parametrize(
        'prefix, expected_columns',
        [(True, ['A_tag Column 1', 'A_tag Column 2', 'A_tag Column 3']),
         (False, ['Column 1 A_tag', 'Column 2 A_tag', 'Column 3 A_tag'])]
    )
    def test_rename_columns_with_sample_column_tags(self, prefix, expected_columns):
        table = self.reader._read_file('table.txt')
        self.reader.column_mapping = {}
        self.reader.column_tag_mapping = {}
        self.reader.sample_column_tags = ['A_tag']
        table = self.reader._rename_columns(table, prefix_tag=prefix)
        assert all([c in table.columns for c in expected_columns])

    @pytest.mark.parametrize(
        'drop_columns, num_dropped_columns',
        [(['Column 1'], 1), (['Column 1', 'Column 2', 'Column 3'], 3),
         ([], 0), (['Column that does not exist'], 0)]
    )
    def test_drop_columns(self, drop_columns, num_dropped_columns):
        table = self.reader._read_file('table.txt')
        table = self.reader._drop_columns(table, drop_columns)
        assert table.shape[1] == self.table_ncolumns - num_dropped_columns

    def test_drop_columns_by_tag(self):
        table = self.reader._read_file('table.txt')
        table = self.reader._drop_columns_by_tag(table, 'A_tag')
        assert all(['A_tag' not in c for c in table.columns])
        assert table.shape[1] < self.table_ncolumns


class TestMQReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.MQReader(
            './tests/testdata/maxquant', contaminant_tag='contam_',
        )

    def test_testdata_setup(self):
        assert os.path.isdir(self.reader.data_directory)

    def test_drop_decoy(self):
        table = self.reader._read_file('proteins')
        table = self.reader._drop_decoy(table)
        is_decoy = table['Majority protein IDs'].str.contains('REV__')
        assert not is_decoy.any()

    def test_drop_idbysite(self):
        table = self.reader._read_file('proteins')
        table = self.reader._drop_idbysite(table)
        is_idbysite = (table['Only identified by site'] == '+')
        assert not is_idbysite.any()

    def test_add_protein_entries(self):
        table = pd.DataFrame({
            'Majority protein IDs': ['B;A;C', 'D', 'E;F', 'G;H;I'],
            'Peptide counts (all)': ['5;5;3', '3', '6;3', '6;6;6'],
        })
        leading_proteins = ['B;A', 'D', 'E', 'G;H;I']
        representative_protein = ['B', 'D', 'E', 'G']
        protein_reported_by_software = representative_protein

        table = self.reader._add_protein_entries(table)
        assert table['Leading proteins'].tolist() == leading_proteins
        assert table['Representative protein'].tolist() == representative_protein
        assert table['Protein reported by software'].tolist() == protein_reported_by_software

    def test_process_protein_entries(self):
        table = pd.DataFrame({
            'Majority protein IDs': [
                'B;A;C', 'D', 'E;F', 'G;H;I', 'CON__x|J|x;J', 'CON__x|K|x'],
            'Peptide counts (all)': ['5;5;3', '3', '6;3', '6;6;6', '4;4', '4'],
        })
        leading_proteins = ['B;A', 'D', 'E', 'G;H;I', 'J;J', 'K']
        representative_protein = ['B', 'D', 'E', 'G', 'J', 'K']
        protein_reported_by_software = representative_protein
        is_contaminant = [False, False, False, False, True, True]
        # TODO: Change after sorting
        # leading_proteins = ['A;B', 'D', 'E', 'G;H;I', 'J;J', 'K']
        # representative_protein = ['A', 'D', 'E', 'G', 'J', 'K']
        # is_contaminant = [False, False, False, False, True, True]

        table = self.reader._process_protein_entries(table)
        assert table['Leading proteins'].tolist() == leading_proteins
        assert table['Representative protein'].tolist() == representative_protein
        assert table['Protein reported by software'].tolist() == protein_reported_by_software
        assert table['Potential contaminant'].tolist() == is_contaminant

    def test_integration_import_proteins(self):
        table = self.reader.import_proteins(
            rename_columns=True,
            prefix_column_tags=False,
            drop_decoy=True,
            drop_idbysite=True,
            sort_proteins=True,
            drop_protein_info=True,
            mark_contaminants=False,  # Not tested
            special_proteins=[],  # Not tested
        )
        assert not (table['Reverse'] == '+').any()
        assert not (table['Only identified by site'] == '+').any()
        assert not table['Representative protein'].str.contains('REV__').any()
        assert 'Total peptides' in table
        assert '12500amol_1 Intensity' in table
        assert not table.columns.str.contains('iBAQ').any()
        assert not table.columns.str.contains('site positions').any()
        assert 'Protein names' not in table.columns
        assert not table['Representative protein'].str.contains('contam_P00330').any()


class TestFPReader:
    @pytest.fixture(autouse=True)
    def _init_reader(self):
        self.reader = msreport.reader.FPReader(
            './tests/testdata/fragpipe', contaminant_tag='contam_',
        )

    def test_testdata_setup(self):
        assert os.path.isdir(self.reader.data_directory)

    def test_add_protein_entries(self):
        table = pd.DataFrame({
            'Protein': [
                'x|B|b', 'x|D|d', 'x|E|e', 'x|G|g'],
            'Indistinguishable Proteins': [
                'x|A|a', '', '', 'x|H|h, x|I|i'],
        })
        leading_proteins = ['B;A', 'D', 'E', 'G;H;I']
        representative_protein = ['B', 'D', 'E', 'G']
        protein_reported_by_software = representative_protein

        table = self.reader._add_protein_entries(table)
        assert table['Leading proteins'].tolist() == leading_proteins
        assert table['Representative protein'].tolist() == representative_protein
        assert table['Protein reported by software'].tolist() == protein_reported_by_software
