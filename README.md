# Enhanced Glass Defect Localization through Polarization Information Fusion and Reciprocal Feature Evolution: New Challenge, Dataset and Baseline

## Abstract
Glass containers are widely used in daily life, but existing defect detection methods are still difficult to detect accurately. Firstly, the extreme scarcity of glass container defect samples makes it difficult to train the model accurately. Secondly, due to the characteristics of transparent reflection, it is challenging to obtain blurred defect separation edge and context information. Finally, the existing positioning loss does not consider the shape accuracy of the predicted box, the accurate positioning information cannot be obtained. This paper introduces polarization and RGB information to create the glass container defect dataset locations containing 60,000+ samples. Subsequently, this paper proposes a novel interactive decoupled feature evolution network (IDFE-Net) by decoupling edge and context information in a feature interactive coevolution method. Finally, with the demand for accurate positioning in industrial defect detection scenarios, this paper proposes a novel Inforced-IoU, which can obtain more precise position information by adaptively adjusting the scale of the predicted box. Experiments show that our method only uses 18.1 GFLOPs and achieves 94.61 % and 67.43 % mAP on glass container and wood defects datasets, better than the current state-of-the-art method.
## GCD data statistical distribution
| Class            | bubble              | oil                   | Plastering thread         | Black spot        | Quenched grain               |
|------------------|---------------------|-----------------------|---------------------------|-------------------|------------------------------|
| S                | 22905               | 3279                  | 2239                      | 19463             | 3104                         |
| M                | 823                 | 2142                  | 3850                      | 219               | 2610                         |
| L                | 572                 | 6489                  | 13356                     | 188               | 2066                         |
| Feature describe | bright round hollow | dark irregular shapes | Irregular striped pattern | dark round shapes | blurred and irregular shapes |

