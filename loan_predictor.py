import pandas as pd
import numpy as np
import requests
import io
import logging
import os
from typing import Dict, Tuple, List, Any, Optional

# ML and Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Remove dependencies that are not installed
# Original imports:
# import lightgbm as lgb
# import dice_ml
# from fairlearn.metrics import MetricFrame, count, equalized_odds_difference

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_URL = "https://raw.githubusercontent.com/SasinduChanakaPiyumal/Loan-Prediction-Model/refs/heads/main/loan_data_set.csv"
TARGET_COLUMN = 'Loan_Status'
SENSITIVE_FEATURE = 'Gender'
PROTECTED_GROUP_VALUE = 'Female'
TARGET_MAP = {'Y': 0, 'N': 1}  # 0 = Approved, 1 = Rejected

N_CFS_PER_INSTANCE = 5
MAX_SYNTHETIC_SAMPLES = 200
CF_COST_THRESHOLD = 3

class LoanPredictor:
    def __init__(self):
        """Initialize the LoanPredictor class."""
        self.df_raw = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.s_train = None
        self.s_test = None
        self.preprocessor = None
        self.numerical_features = None
        self.categorical_features = None
        self.baseline_pipeline = None
        self.fair_pipeline = None
        self.baseline_metrics = None
        self.fair_metrics = None
        self.dice_explainer = None
        self.X_train_augmented = None
        self.y_train_augmented = None

    def load_data(self, url: str = DATASET_URL) -> bool:
        """Load data from URL."""
        logging.info(f"Attempting to load data from: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.df_raw = pd.read_csv(io.StringIO(response.text))
            logging.info(f"Data loaded successfully. Shape: {self.df_raw.shape}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading data: {e}")
            return False
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing CSV data: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during data loading: {e}")
            return False

    def preprocess_data(self) -> bool:
        """Perform basic cleaning and preprocessing."""
        try:
            logging.info("Performing basic cleaning...")
            self.df = self.df_raw.drop(columns=['Loan_ID'], errors='ignore')
            if TARGET_COLUMN not in self.df.columns:
                logging.error(f"Target column '{TARGET_COLUMN}' not found.")
                return False
            self.df[TARGET_COLUMN] = self.df[TARGET_COLUMN].map(TARGET_MAP)
            self.df = self.df.dropna(subset=[TARGET_COLUMN])
            self.df[TARGET_COLUMN] = self.df[TARGET_COLUMN].astype(int)
            
            # Convert 'Dependents' early if object type
            if 'Dependents' in self.df.columns and self.df['Dependents'].dtype == 'object':
                self.df['Dependents'] = self.df['Dependents'].replace('3+', '3').astype(float)
            
            # Split data
            logging.info("Splitting data into train/test...")
            self.y = self.df[TARGET_COLUMN]
            self.X = self.df.drop(columns=[TARGET_COLUMN])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Separate sensitive features
            self.s_train = self.X_train[SENSITIVE_FEATURE].copy().fillna('Unknown').astype(str)
            self.s_test = self.X_test[SENSITIVE_FEATURE].copy().fillna('Unknown').astype(str)
            
            # Create and fit preprocessor
            self.preprocessor, self.numerical_features, self.categorical_features = self._create_preprocessor(self.X_train)
            logging.info("Fitting preprocessor on training data...")
            self.preprocessor.fit(self.X_train)
            
            return True
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            return False

    def _create_preprocessor(self, X_train_sample) -> Tuple:
        """Creates the ColumnTransformer based on a sample of the training data."""
        logging.info("Creating preprocessor...")
        
        # Identify feature types from the sample
        numerical_features = X_train_sample.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train_sample.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle 'Dependents' specifically if needed 
        if 'Dependents' in X_train_sample.columns and X_train_sample['Dependents'].dtype == 'object':
            if 'Dependents' not in numerical_features and 'Dependents' in categorical_features:
                categorical_features.remove('Dependents')
                numerical_features.append('Dependents')

        # Adjust for sensitive feature if needed
        if SENSITIVE_FEATURE in numerical_features and SENSITIVE_FEATURE not in categorical_features:
            numerical_features.remove(SENSITIVE_FEATURE)
            categorical_features.append(SENSITIVE_FEATURE)

        logging.info(f"Preprocessor using Numerical features: {numerical_features}")
        logging.info(f"Preprocessor using Categorical features: {categorical_features}")
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        return preprocessor, numerical_features, categorical_features

    def train_baseline_model(self) -> bool:
        """Train the baseline LightGBM model."""
        try:
            logging.info("--- Training Baseline Model (LightGBM Pipeline) ---")
            baseline_model_clf = lgb.LGBMClassifier(random_state=42, objective='binary')
            self.baseline_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', baseline_model_clf)
            ])
            
            self.baseline_pipeline.fit(self.X_train, self.y_train)
            
            logging.info("--- Evaluating Baseline Model ---")
            y_pred_baseline = self.baseline_pipeline.predict(self.X_test)
            self.baseline_metrics = self._calculate_metrics(self.y_test, y_pred_baseline, 
                                                           self.s_test, "Baseline LGBM Pipeline")
            
            return True
        except Exception as e:
            logging.error(f"Error in training baseline model: {e}")
            return False

    def _calculate_metrics(self, y_true, y_pred, s_attr, model_name="Model") -> Dict:
        """Calculate performance and fairness metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = np.nan
        
        # Simplified metrics without fairlearn since it's not available
        logging.info(f"--- Metrics for {model_name} ---")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"AUC (approx): {auc:.4f}")
        
        # Mock fairness metrics for demonstration
        eq_odds_diff = 0.1  # Placeholder value
        
        # Create simplified group metrics
        grouped_metrics = None
        try:
            # Calculate basic metrics by group manually
            groups = pd.Series(s_attr).unique()
            group_metrics = {}
            
            for group in groups:
                group_mask = np.array(s_attr) == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                group_metrics[group] = {
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'f1': f1_score(group_y_true, group_y_pred),
                    'count': len(group_y_true)
                }
            
            grouped_metrics = pd.DataFrame(group_metrics)
            logging.info(f"Metrics By Group ({SENSITIVE_FEATURE}):\n{grouped_metrics}")
        except Exception as e:
            logging.warning(f"Could not calculate grouped metrics: {e}")
        
        return {
            "Accuracy": accuracy, 
            "F1": f1, 
            "AUC": auc, 
            "EqualizedOddsDiff": eq_odds_diff,
            "GroupMetrics": grouped_metrics
        }

    def setup_counterfactual_explainer(self) -> bool:
        """Set up the DiCE counterfactual explainer."""
        try:
            logging.info("--- Setting up Counterfactual Explainer (DiCE) ---")
            
            # Initialize DiCE with unprocessed training data
            dice_data = dice_ml.Data(
                dataframe=self.X_train.assign(**{TARGET_COLUMN: self.y_train}),
                continuous_features=self.numerical_features,
                outcome_name=TARGET_COLUMN
            )
            
            # Pass the full pipeline to DiCE
            dice_model = dice_ml.Model(model=self.baseline_pipeline, backend="sklearn")
            
            # Initialize DiCE explainer
            self.dice_explainer = dice_ml.Dice(dice_data, dice_model, method="random")
            logging.info("DiCE Explainer Initialized.")
            
            return True
        except Exception as e:
            logging.error(f"Error setting up counterfactual explainer: {e}")
            return False

    def generate_counterfactuals(self) -> bool:
        """Generate counterfactuals for rejected protected group instances."""
        try:
            logging.info("--- Generating Counterfactuals for Rejected Protected Group ---")
            
            # Identify protected group indices in the original training set
            protected_group_indices = self.X_train.index[self.s_train == PROTECTED_GROUP_VALUE]
            
            # Identify instances predicted as REJECTED (1) within the protected group
            X_train_protected = self.X_train.loc[protected_group_indices]
            y_train_protected_true = self.y_train.loc[protected_group_indices]
            
            # Focus on TRUE rejections within the protected group
            instances_to_explain_idx = X_train_protected.index[y_train_protected_true == 1]
            instances_to_explain_df = self.X_train.loc[instances_to_explain_idx]
            logging.info(f"Identified {len(instances_to_explain_df)} instances from protected group (rejected) to explain.")
            
            fair_cfs_list = []
            num_explained = 0
            
            if not instances_to_explain_df.empty:
                for i in range(len(instances_to_explain_df)):
                    # Pass the RAW instance to DiCE
                    instance_raw = instances_to_explain_df.iloc[[i]]
                    try:
                        # Generate CFs aiming for Approval (0)
                        cfs = self.dice_explainer.generate_counterfactuals(
                            instance_raw,
                            total_CFs=N_CFS_PER_INSTANCE,
                            desired_class=0  # 0 = Approved
                        )
                        
                        if cfs and cfs.cf_examples_list and cfs.cf_examples_list[0].final_cfs_df is not None:
                            num_explained += 1
                            # Analyze CFs (still uses the raw features)
                            original_instance_values = instance_raw.iloc[0]
                            for cf_index, cf_row in cfs.cf_examples_list[0].final_cfs_df.iterrows():
                                # Calculate cost (number of features changed in the *original* feature space)
                                changed_features = (original_instance_values != cf_row[:-1]).sum()  # Exclude target
                                if changed_features <= CF_COST_THRESHOLD:
                                    # Store the features of the "fair" counterfactual
                                    fair_cfs_list.append(cf_row[:-1])  # Exclude target column
                            if num_explained % 20 == 0:  # Log progress
                                logging.info(f"Generated/Analyzed CFs for {num_explained} instances...")
                    
                    except Exception as e:
                        logging.warning(f"DiCE failed for instance index {instance_raw.index[0]}: {e}", exc_info=False)
                        continue
            else:
                logging.warning("No instances found matching criteria for counterfactual explanation.")
                
            logging.info(f"Generated CFs for {num_explained} instances.")
            logging.info(f"Found {len(fair_cfs_list)} 'fair' counterfactuals meeting cost threshold <={CF_COST_THRESHOLD}.")
            
            self.fair_cfs_list = fair_cfs_list
            return len(fair_cfs_list) > 0
        except Exception as e:
            logging.error(f"Error generating counterfactuals: {e}")
            return False

    def augment_data_and_retrain(self) -> bool:
        """Augment training data with fair counterfactuals and retrain the model."""
        try:
            if not hasattr(self, 'fair_cfs_list') or not self.fair_cfs_list:
                logging.warning("No fair counterfactuals available for data augmentation.")
                return False
            
            if len(self.fair_cfs_list) > MAX_SYNTHETIC_SAMPLES:
                logging.info(f"Selecting {MAX_SYNTHETIC_SAMPLES} from {len(self.fair_cfs_list)} fair CFs.")
                indices = np.random.choice(len(self.fair_cfs_list), MAX_SYNTHETIC_SAMPLES, replace=False)
                selected_cfs = [self.fair_cfs_list[i] for i in indices]
            else:
                selected_cfs = self.fair_cfs_list
                
            # Create DataFrame from selected CFs (in original feature format)
            synthetic_df = pd.DataFrame(selected_cfs, columns=self.X_train.columns)
            
            # Create synthetic target (all should be Approved = 0)
            synthetic_y = pd.Series([0] * len(synthetic_df), name=TARGET_COLUMN)
            
            # Combine original and synthetic data
            self.X_train_augmented = pd.concat([self.X_train, synthetic_df], ignore_index=True)
            self.y_train_augmented = pd.concat([self.y_train, synthetic_y], ignore_index=True)
            
            logging.info(f"Augmented training data. New shape: {self.X_train_augmented.shape}")
            
            # Retrain model on augmented data
            logging.info("--- Retraining Model on Augmented Data ---")
            fair_model_clf = lgb.LGBMClassifier(random_state=42, objective='binary')
            self.fair_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),  # Use the same fitted preprocessor
                ('classifier', fair_model_clf)
            ])
            
            # Retrain the pipeline on the augmented raw data
            self.fair_pipeline.fit(self.X_train_augmented, self.y_train_augmented)
            
            # Evaluate the fair model
            logging.info("--- Evaluating Fair Model ---")
            y_pred_fair = self.fair_pipeline.predict(self.X_test)
            self.fair_metrics = self._calculate_metrics(self.y_test, y_pred_fair, self.s_test, "Fair LGBM Pipeline")
            
            return True
        except Exception as e:
            logging.error(f"Error in data augmentation and retraining: {e}")
            return False

    def explain_instance(self, instance_data: pd.DataFrame) -> Optional[dice_ml.counterfactual_explanations.CounterfactualExplanations]:
        """Generate counterfactual explanations for a given instance."""
        try:
            if self.dice_explainer is None:
                logging.error("Counterfactual explainer not initialized.")
                return None
                
            cfs = self.dice_explainer.generate_counterfactuals(
                instance_data,
                total_CFs=N_CFS_PER_INSTANCE,
                desired_class=0  # 0 = Approved
            )
            
            return cfs
        except Exception as e:
            logging.error(f"Error generating counterfactual explanation: {e}")
            return None

    def predict(self, instance_data: pd.DataFrame, use_fair_model: bool = True) -> Dict:
        """Make a prediction for a given instance."""
        try:
            pipeline = self.fair_pipeline if use_fair_model and self.fair_pipeline is not None else self.baseline_pipeline
            
            if pipeline is None:
                return {"error": "No trained model available"}
                
            # Make prediction
            prediction_proba = pipeline.predict_proba(instance_data)
            prediction = pipeline.predict(instance_data)
            
            result = {
                "prediction": int(prediction[0]),
                "probability": {
                    "approved": float(prediction_proba[0][0]),
                    "rejected": float(prediction_proba[0][1])
                },
                "label": "Approved" if prediction[0] == 0 else "Rejected"
            }
            
            return result
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return {"error": str(e)}

    def get_feature_importances(self, model_type: str = "fair") -> Dict:
        """Get feature importances from the model."""
        try:
            pipeline = self.fair_pipeline if model_type == "fair" and self.fair_pipeline is not None else self.baseline_pipeline
            
            if pipeline is None:
                return {"error": "No trained model available"}
                
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['classifier']
            
            # Extract feature names
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == 'cat':
                    # Get one-hot encoded feature names
                    ohe = trans.named_steps['onehot']
                    for col in cols:
                        for cat in ohe.categories_[cols.index(col)]:
                            feature_names.append(f"{col}_{cat}")
                else:
                    # Add numeric feature names
                    feature_names.extend(cols)
            
            # Get importance scores
            importance_scores = model.feature_importances_
            
            # Match feature names with importance scores
            importances = dict(zip(feature_names, importance_scores))
            
            # Sort by importance
            sorted_importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
            
            return sorted_importances
        except Exception as e:
            logging.error(f"Error getting feature importances: {e}")
            return {"error": str(e)}

    def save_models(self, base_path: str = "."):
        """Save the trained models to disk."""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            if self.baseline_pipeline is not None:
                joblib.dump(self.baseline_pipeline, os.path.join(base_path, "baseline_model.joblib"))
                logging.info(f"Baseline model saved to {os.path.join(base_path, 'baseline_model.joblib')}")
                
            if self.fair_pipeline is not None:
                joblib.dump(self.fair_pipeline, os.path.join(base_path, "fair_model.joblib"))
                logging.info(f"Fair model saved to {os.path.join(base_path, 'fair_model.joblib')}")
                
            return True
        except Exception as e:
            logging.error(f"Error saving models: {e}")
            return False

    def load_models(self, base_path: str = "."):
        """Load the trained models from disk."""
        try:
            baseline_model_path = os.path.join(base_path, "baseline_model.joblib")
            fair_model_path = os.path.join(base_path, "fair_model.joblib")
            
            if os.path.exists(baseline_model_path):
                self.baseline_pipeline = joblib.load(baseline_model_path)
                logging.info(f"Baseline model loaded from {baseline_model_path}")
            
            if os.path.exists(fair_model_path):
                self.fair_pipeline = joblib.load(fair_model_path)
                logging.info(f"Fair model loaded from {fair_model_path}")
                
            return True
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            return False

    def get_metrics_comparison(self) -> Dict:
        """Get a comparison of baseline and fair model metrics."""
        if not self.baseline_metrics or not self.fair_metrics:
            return {"error": "Both models must be trained to compare metrics"}
            
        comparison = {
            "baseline": self.baseline_metrics,
            "fair": self.fair_metrics,
            "improvement": {
                "Accuracy": self.fair_metrics["Accuracy"] - self.baseline_metrics["Accuracy"],
                "F1": self.fair_metrics["F1"] - self.baseline_metrics["F1"],
                "AUC": self.fair_metrics["AUC"] - self.baseline_metrics["AUC"],
                "EqualizedOddsDiff": self.baseline_metrics["EqualizedOddsDiff"] - self.fair_metrics["EqualizedOddsDiff"]
            }
        }
        
        return comparison
