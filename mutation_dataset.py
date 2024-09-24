from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

class MutationDataset(Dataset):
    def __init__(
        self, 
        mutations: pd.DataFrame, 
        sample_col: str = 'Tumor_Sample_Barcode',
        gene_col: str = 'Hugo_Symbol',
        variant_col: str = 'Variant_Classification',
        samples: Optional[pd.DataFrame] = None,
        variants: Optional[list[str]] = None,
        genes: Optional[list[str]] = None,
        dense: bool = False
    ) -> None:
        super().__init__()

        self.mutations = mutations
        self.samples = mutations.set_index(sample_col, drop=False)[[sample_col]].drop_duplicates() if samples.empty else samples
        self.variants = variants or mutations[variant_col].unique().tolist()
        self.genes = genes or mutations[gene_col].unique().tolist()

        self.sample_col = sample_col
        self.gene_col = gene_col
        self.variant_col = variant_col
        
        self.simplified_mutations = self.mutations[[sample_col, gene_col, variant_col]]
        self.variant_ids = {variant: idx for idx, variant in enumerate(self.variants)}
        self.gene_ids = {gene: idx for idx, gene in enumerate(self.genes)}

        self.simplified_mutations.replace({gene_col: self.gene_ids, variant_col: self.variant_ids}, inplace=True)
        self.simplified_mutations.set_index(sample_col, inplace=True)

        self.dense = dense


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample: int | str) -> torch.Tensor:
        if isinstance(sample, int):
            sample = self._get_sample_for_index(sample)
        sample_mutations = self._get_simplified_mutations_for_sample(sample)
        output = torch.sparse_coo_tensor(sample_mutations.values.T, [1.]*len(sample_mutations), (len(self.genes), len(self.variants)))

        if self.dense:
            output = output.to_dense()

        return output

    def _get_sample_for_index(self, index: int) -> str:
        return self.samples.index[index]

    def _get_simplified_mutations_for_sample(self, sample_id: str) -> pd.DataFrame:
        if sample_id not in self.simplified_mutations.index:
            return pd.DataFrame(columns=[self.gene_col, self.variant_col], dtype=int)
        
        return self.simplified_mutations.loc[[sample_id], :]
    
    def _get_simplified_mutations_for_tensor(self, mutation_tensor: torch.Tensor) -> pd.DataFrame:
        mutation_array = mutation_tensor.coalesce().indices().numpy().T
        return pd.DataFrame(mutation_array, columns=[self.gene_col, self.variant_col])
    
    def _unsimplify_mutations(self, simplified_mutations: pd.DataFrame) -> pd.DataFrame:
        reverse_gene_ids = {idx: gene for gene, idx in self.gene_ids.items()}
        reverse_variant_ids = {idx: variant for variant, idx in self.variant_ids.items()}

        return simplified_mutations.replace({self.gene_col: reverse_gene_ids, self.variant_col: reverse_variant_ids})


    def get_mutations(self, sample: int | str | torch.Tensor, simplified: bool = False) -> pd.DataFrame:
        if isinstance(sample, int):
            sample = self._get_sample_for_index(sample)
        if isinstance(sample, torch.Tensor):
            mutations = self._get_simplified_mutations_for_tensor(sample)
        else:
            mutations = self._get_simplified_mutations_for_sample(sample)
    
        if not simplified:
            mutations = self._unsimplify_mutations(mutations)

        return mutations
