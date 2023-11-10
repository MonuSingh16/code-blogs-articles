### Popular Cross Validation Techniques

Tuning and validating ML models on a single validation set is often misleading (such as single train - test split), as they
are easy to implement but have optimistic results. The lucky random split leads to model perform exceptionally well on
validation set but poorly on unseen or new data

CV involves repeatedly partitioning data into subsets, training models on few subset and validating on remaning. it provides more 
robust and unbiased estimate of model performance

#### Leave-One-Out Cross-Validation

- Leave one data point for validation.
- Train the model on the remaining data points.
- Repeat for all points.
- This is practically infeasible when you have tons of data points. This is because number of models is equal to number of data points.

We can extend this to Leave-p-Out Cross-Validation, where, in each iteration, p observations are reserved for validation and the rest are used for training.

#### K-Fold Cross-Validation

- Split data into k equally-sized subsets.
- Select one subset for validation.
- Train the model on the remaining subsets.
- Repeat for all subsets

#### Rolling Cross-Validation

- Mostly used for data with temporal structure.
- Data splitting respects the temporal order, using a fixed-size training window.
- The model is evaluated on the subsequent window

#### Blocked Cross-Validation

- Another common technique for time-series data.
- In contrast to rolling cross-validation, the slice of data is intentionally kept short if the variance does not change appreciably from one window to the next.
- This also saves computation over rolling cross-validation

#### Stratified Cross-Validation

- The above techniques may not work for imbalanced datasets. Thus, this technique is mostly used for preserving the class distribution.
- The partitioning ensures that the class distribution is preserved.