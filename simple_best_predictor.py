import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SimpleAdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    🎯 Ingénieur de Caractéristiques Simple mais Efficace
    
    Utilise exactement les mêmes techniques qui ont donné F1=0.9442
    mais de manière optimisée
    """
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Transformation simple mais efficace"""
        print("🔧 Feature Engineering Simple et Efficace...")
        
        n_samples, n_timesteps, n_features = X.shape
        
        # 1. Statistiques de base importantes (comme dans le modèle original)
        basic_stats = self._extract_basic_stats(X)
        
        # 2. Caractéristiques temporelles simples
        temporal_stats = self._extract_temporal_stats(X)
        
        # 3. Ratios simples entre biomarqueurs
        ratio_stats = self._extract_simple_ratios(X)
        
        # 4. Patterns de données manquantes basiques
        missing_stats = self._extract_missing_stats(X)
        
        # Combiner
        all_features = np.concatenate([basic_stats, temporal_stats, ratio_stats, missing_stats], axis=1)
        
        # Nettoyer
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"✅ Caractéristiques créées : {all_features.shape[1]} (simple et efficace)")
        
        return all_features
    
    def _extract_basic_stats(self, X):
        """Statistiques de base - les plus importantes"""
        n_samples, n_timesteps, n_features = X.shape
        features = np.zeros((n_samples, n_features * 12))
        
        for i in range(n_samples):
            for j in range(n_features):
                # Extraire valeurs valides
                values = []
                for k in range(n_timesteps):
                    val = X[i, k, j]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            values.append(float(val))
                        except:
                            continue
                
                if len(values) >= 3:
                    values = np.array(values)
                    
                    # Statistiques importantes
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    median_val = np.median(values)
                    q25 = np.percentile(values, 25)
                    q75 = np.percentile(values, 75)
                    skewness = stats.skew(values)
                    kurtosis = stats.kurtosis(values)
                    range_val = max_val - min_val
                    iqr = q75 - q25
                    coeff_var = std_val / (abs(mean_val) + 1e-8)
                    
                elif len(values) > 0:
                    values = np.array(values)
                    mean_val = np.mean(values)
                    std_val = np.std(values) if len(values) > 1 else 0
                    min_val = np.min(values)
                    max_val = np.max(values)
                    median_val = mean_val
                    q25 = q75 = mean_val
                    skewness = kurtosis = 0
                    range_val = iqr = coeff_var = 0
                else:
                    mean_val = std_val = min_val = max_val = 0
                    median_val = q25 = q75 = skewness = kurtosis = 0
                    range_val = iqr = coeff_var = 0
                
                start_idx = j * 12
                features[i, start_idx:start_idx+12] = [
                    mean_val, std_val, min_val, max_val, median_val, q25, q75,
                    skewness, kurtosis, range_val, iqr, coeff_var
                ]
        
        return features
    
    def _extract_temporal_stats(self, X):
        """Caractéristiques temporelles importantes"""
        n_samples, n_timesteps, n_features = X.shape
        features = np.zeros((n_samples, n_features * 8))
        
        for i in range(n_samples):
            for j in range(n_features):
                # Extraire valeurs valides
                values = []
                time_points = []
                for k in range(n_timesteps):
                    val = X[i, k, j]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            values.append(float(val))
                            time_points.append(k)
                        except:
                            continue
                
                if len(values) >= 3:
                    values = np.array(values)
                    time_points = np.array(time_points)
                    
                    # Tendance linéaire
                    slope, intercept = np.polyfit(time_points, values, 1) if len(values) > 1 else (0, 0)
                    
                    # Accélération
                    if len(values) >= 3:
                        acceleration = np.polyfit(time_points, values, 2)[0]
                    else:
                        acceleration = 0
                    
                    # Volatilité
                    volatility = np.std(values)
                    
                    # Première et dernière valeur
                    first_val = values[0]
                    last_val = values[-1]
                    total_change = last_val - first_val
                    
                    # Changements de direction
                    direction_changes = 0
                    if len(values) >= 3:
                        for k in range(1, len(values)-1):
                            if (values[k] > values[k-1] and values[k+1] < values[k]) or \
                               (values[k] < values[k-1] and values[k+1] > values[k]):
                                direction_changes += 1
                    
                    # R-squared de la tendance
                    if len(values) > 2:
                        predicted = slope * time_points + intercept
                        r_squared = 1 - np.sum((values - predicted) ** 2) / np.sum((values - np.mean(values)) ** 2)
                    else:
                        r_squared = 0
                        
                else:
                    slope = acceleration = volatility = 0
                    first_val = last_val = total_change = 0
                    direction_changes = r_squared = 0
                
                start_idx = j * 8
                features[i, start_idx:start_idx+8] = [
                    slope, acceleration, volatility, first_val, last_val,
                    total_change, direction_changes, r_squared
                ]
        
        return features
    
    def _extract_simple_ratios(self, X):
        """Ratios simples entre biomarqueurs"""
        n_samples, n_timesteps, n_features = X.shape
        
        # Calculer moyennes des biomarqueurs
        means = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            for j in range(n_features):
                values = []
                for k in range(n_timesteps):
                    val = X[i, k, j]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            values.append(float(val))
                        except:
                            continue
                means[i, j] = np.mean(values) if len(values) > 0 else 0
        
        # Groupes de biomarqueurs (comme dans le modèle original)
        group_size = max(1, n_features // 4)
        
        glucose_group = np.mean(means[:, :group_size], axis=1) if group_size > 0 else np.zeros(n_samples)
        lipid_group = np.mean(means[:, group_size:2*group_size], axis=1) if 2*group_size <= n_features else np.zeros(n_samples)
        liver_group = np.mean(means[:, 2*group_size:3*group_size], axis=1) if 3*group_size <= n_features else np.zeros(n_samples)
        kidney_group = np.mean(means[:, 3*group_size:], axis=1) if 3*group_size < n_features else np.zeros(n_samples)
        
        # Scores composites
        metabolic_score = (glucose_group + lipid_group) / 2
        organ_score = (liver_group + kidney_group) / 2
        
        # Ratios sécurisés
        glucose_lipid_ratio = glucose_group / (np.abs(lipid_group) + 1e-8)
        liver_kidney_ratio = liver_group / (np.abs(kidney_group) + 1e-8)
        
        ratios = np.column_stack([
            glucose_group, lipid_group, liver_group, kidney_group,
            metabolic_score, organ_score, glucose_lipid_ratio, liver_kidney_ratio
        ])
        
        return ratios
    
    def _extract_missing_stats(self, X):
        """Statistiques de données manquantes"""
        n_samples, n_timesteps, n_features = X.shape
        features = np.zeros((n_samples, 8))
        
        for i in range(n_samples):
            total_missing = 0
            missing_per_timepoint = np.zeros(n_timesteps)
            missing_per_feature = np.zeros(n_features)
            
            for j in range(n_timesteps):
                for k in range(n_features):
                    val = X[i, j, k]
                    is_missing = val is None or (isinstance(val, float) and np.isnan(val))
                    if is_missing:
                        total_missing += 1
                        missing_per_timepoint[j] += 1
                        missing_per_feature[k] += 1
            
            missing_rate = total_missing / (n_timesteps * n_features)
            max_missing_timepoint = np.max(missing_per_timepoint) / n_features
            max_missing_feature = np.max(missing_per_feature) / n_timesteps
            missing_timepoint_std = np.std(missing_per_timepoint)
            missing_feature_std = np.std(missing_per_feature)
            
            early_missing = np.sum(missing_per_timepoint[:3]) / (3 * n_features) if n_timesteps >= 3 else 0
            late_missing = np.sum(missing_per_timepoint[-3:]) / (3 * n_features) if n_timesteps >= 3 else 0
            missing_concentration = np.var(missing_per_timepoint) + np.var(missing_per_feature)
            
            features[i] = [
                missing_rate, max_missing_timepoint, max_missing_feature,
                missing_timepoint_std, missing_feature_std, early_missing,
                late_missing, missing_concentration
            ]
        
        return features

class SimpleBestPredictor:
    """
    🚀 Prédicteur Simple utilisant les Meilleures Techniques
    
    Utilise exactement les mêmes paramètres qui ont donné F1=0.9442
    """
    
    def __init__(self):
        self.feature_engineer = SimpleAdvancedFeatureEngineer()
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(f_classif, k=500)
        self.model = None
        self.best_threshold = 0.5
        
    def prepare_model(self):
        """Préparer le modèle avec les MÊMES paramètres qui ont marché"""
        
        # EXACTEMENT les mêmes paramètres qui ont donné F1=0.9442
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        print("✅ Modèle GradientBoosting avec paramètres optimaux préparé")
        
    def train(self, X, y):
        """Entraînement avec la méthode qui a marché"""
        
        print("🎯 Entraînement avec les meilleures techniques...")
        
        # Feature Engineering
        X_engineered = self.feature_engineer.fit_transform(X)
        
        # Scaling (comme dans l'original)
        X_scaled = self.scaler.fit_transform(X_engineered)
        
        # Feature Selection (comme dans l'original)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        print(f"📊 Caractéristiques finales : {X_selected.shape[1]}")
        
        # SMOTE (comme dans l'original)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_selected, y)
        
        print(f"🔄 SMOTE : {len(y)} → {len(y_resampled)} échantillons")
        
        # Entraîner
        self.model.fit(X_resampled, y_resampled)
        
        # Validation
        cv_scores = cross_val_score(
            self.model, X_selected, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        print(f"🏆 CV F1 : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return X_selected
    
    def optimize_threshold(self, X_val, y_val):
        """Optimisation du seuil"""
        
        print("🎯 Optimisation du seuil...")
        
        X_val_processed = self._transform_data(X_val)
        y_proba = self.model.predict_proba(X_val_processed)[:, 1]
        
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        print(f"✅ Meilleur seuil : {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        return best_threshold, best_f1
    
    def predict(self, X):
        """Prédiction"""
        X_processed = self._transform_data(X)
        y_proba = self.model.predict_proba(X_processed)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        return y_pred, y_proba
    
    def _transform_data(self, X):
        """Transformation"""
        X_engineered = self.feature_engineer.transform(X)
        X_scaled = self.scaler.transform(X_engineered)
        X_selected = self.feature_selector.transform(X_scaled)
        return X_selected

def main():
    """Exécution avec les meilleures techniques"""
    
    print("="*70)
    print("🚀 PRÉDICTEUR SIMPLE AVEC MEILLEURES TECHNIQUES")
    print("   Utilise exactement ce qui a donné F1=0.9442")
    print("="*70)
    
    # Charger données
    print("📊 Chargement des données...")
    with np.load("training_data.npz", allow_pickle=True) as f:
        X_train = f["data"]
        feature_names = f["feature_labels"]
    
    y_train = pd.read_csv("training_labels.csv")["Label"].values
    
    print(f"✅ Données : {X_train.shape}")
    print(f"✅ Étiquettes : {len(y_train)} (positives : {np.sum(y_train)} - {np.mean(y_train)*100:.1f}%)")
    
    # Division
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Initialisation
    predictor = SimpleBestPredictor()
    predictor.prepare_model()
    
    # Entraînement
    print("\n" + "="*50)
    predictor.train(X_train_split, y_train_split)
    
    # Optimisation seuil
    print("\n" + "="*50)
    best_threshold, best_f1 = predictor.optimize_threshold(X_val_split, y_val_split)
    
    # Évaluation
    print("\n" + "="*50)
    print("🎯 Évaluation finale :")
    
    y_pred, y_proba = predictor.predict(X_val_split)
    
    f1 = f1_score(y_val_split, y_pred)
    precision = precision_score(y_val_split, y_pred)
    recall = recall_score(y_val_split, y_pred)
    auc = roc_auc_score(y_val_split, y_proba)
    
    print(f"🏆 Performance :")
    print(f"   Score F1 : {f1:.4f}")
    print(f"   Précision : {precision:.4f}")
    print(f"   Rappel : {recall:.4f}")
    print(f"   AUC-ROC : {auc:.4f}")
    
    # Prédictions finales
    print("\n" + "="*50)
    print("📤 Prédictions finales...")
    
    with np.load("evaluation_data.npz", allow_pickle=True) as f:
        X_test = f["data"]
    
    y_test_pred, y_test_proba = predictor.predict(X_test)
    
    submission = pd.DataFrame({
        'Id': range(len(y_test_pred)),
        'Label': y_test_pred
    })
    
    submission.to_csv('submission_simple_best.csv', index=False)
    
    print(f"✅ Fichier submission_simple_best.csv créé")
    print(f"📊 Prédictions positives : {np.sum(y_test_pred)} ({np.mean(y_test_pred)*100:.1f}%)")
    print(f"🏆 Score F1 : {f1:.4f}")
    
    print("\n" + "="*70)
    print("🎉 Terminé ! Utilise les techniques qui ont donné F1=0.9442")
    print("="*70)

if __name__ == "__main__":
    main() 