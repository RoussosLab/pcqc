#First do Python QC Check

library(Seurat)
df = read.table(file = 'dense_umis.tsv', sep = '\t', header = TRUE, row.names = 1)
#df = read.table(file = 'subset_dense_umis.csv', sep = ',', header = TRUE, row.names = 1)
pbmc.htos <- as.matrix(df[1:4,]) #confirm with rownames that you have the right rows
#normally this should actually include counts- but not necessary
pbmc.hashtag <- CreateSeuratObject(counts = pbmc.htos)
pbmc.hashtag[["HTO"]] <- CreateAssayObject(counts = pbmc.htos)
pbmc.hashtag <- NormalizeData(pbmc.hashtag, assay = "HTO", normalization.method = "CLR")
pbmc.hashtag <- HTODemux(pbmc.hashtag, assay = "HTO", positive.quantile = 0.99)
table(pbmc.hashtag$HTO_classification.global)

# Doublet Negative  Singlet
# 2184        1     8255
pdfPath = "sc_pcqc/hto.pdf"
pdf(file = pdfPath)
Idents(pbmc.hashtag) <- "HTO_maxID"
RidgePlot(pbmc.hashtag, assay = "HTO", features = rownames(pbmc.hashtag[["HTO"]])[1:4], ncol = 4)
FeatureScatter(pbmc.hashtag, feature1 = "hto_HTO1-GTCAACTCTTTAGCG", feature2 = "hto_HTO2-TGATGGCCTATTGGG")
FeatureScatter(pbmc.hashtag, feature1 = "hto_HTO3-TTCCGCCTCTCTTTG", feature2 = "hto_HTO4-AGTAAGTTCAGCGTA")
FeatureScatter(pbmc.hashtag, feature1 = "hto_HTO3-TTCCGCCTCTCTTTG", feature2 = "hto_HTO2-TGATGGCCTATTGGG")
HTOHeatmap(pbmc.hashtag, assay = "HTO", ncells = 5000)
dev.off()
hto_list <- as.list(pbmc.hashtag$HTO_classification.global)
hto_df <- do.call("rbind", lapply(hto_list, as.data.frame))
write.csv(hto_df, file = "sc_pcqc/hto.csv")

#see https://satijalab.org/seurat/v3.2/hashing_vignette.html for more plotting options
