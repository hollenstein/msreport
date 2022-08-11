#!/usr/bin/env Rscript
library(limma)


.two_group_limma <- function(data_frame, column_groups, group1, group2, trend) {
    design <- model.matrix(~factor(column_groups, levels=c(group1, group2)))
    colnames(design) <- c(group1, group2)

    fit_lm <- lmFit(data_frame, design)
    fit_ebayes <- eBayes(fit_lm, trend = trend)
    limma_results <- topTable(fit_ebayes, number = Inf, coef = group2,
                              adjust = "BH", sort.by = "none")
    limma_results <- cbind(rownames(limma_results), limma_results)
    colnames(limma_results)[1] <- "id"
    return(limma_results)
}
