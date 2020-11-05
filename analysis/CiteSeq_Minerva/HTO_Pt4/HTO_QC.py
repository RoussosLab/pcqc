import pandas as pd
barcodes_final = pd.read_csv('sc_pcqc/adt_cdna_barcode_match.csv')
hto = pd.read_csv('dense_umis.tsv', sep = '\t')
valid_cols = hto.columns
barcodes_hto = barcodes_final.loc[barcodes_final['barcode_adt'].isin(valid_cols)]
barcodes_hto['final_filter'].sum()
barcodes_hto['final_filter'].sum()/barcodes_final['final_filter'].sum()

hto_df = hto.transpose()
hto_df.columns = hto['Unnamed: 0'].unique()
hto_df.head()
hto_df['barcode_adt'] = hto_df.index
valid_barcodes = barcodes_hto.loc[barcodes_hto['final_filter'] == True]
hto_final = pd.merge(hto_df, valid_barcodes[['barcode_adt']], on = 'barcode_adt')
hto_final.head()
hto_final.index = hto_final['barcode_adt']
hto_final.drop(columns = 'barcode_adt', inplace = True)
hto_final.head()
hto_final.transpose().to_csv('subset_dense_umis.csv')

#After HTO clustering R analysis
import pandas as pd
hto_df = pd.read_csv('sc_pcqc/hto.csv')
hto_df.columns = ['barcode','value']
merged = pd.merge(barcodes_final, hto_df, how = 'inner', left_on = 'simple_barcode', right_on = 'barcode')
merged.loc[merged['final_filter'] == True,'value'].value_counts()
