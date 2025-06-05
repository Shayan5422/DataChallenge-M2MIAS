import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

class SupremeFinalEnsemble:
    def __init__(self):
        self.all_submissions = {}
        self.scores = {}
        
    def load_all_submissions(self):
        """Load ALL available submissions including our new ensembles"""
        
        # All submissions with their scores (including our new ensembles)
        all_files = {
            # Original high performers
            'ensemble_weighted_optimized.csv': 0.47130,  # Current champion
            'submissionV2.csv': 0.46954,
            'submissionV2 (3).csv': 0.46889,
            'submission_ensemble.csv': 0.46748,
            'submission_weighted_blend.csv': 0.46745,
            'evaluation_predictions_caro.csv': 0.46716,
            'submissionV3_ontology.csv': 0.46292,
            'submissionV2 (2).csv': 0.46064,
            'evaluation_predictions (1).csv': 0.46055,
            
            # Our new advanced ensembles (estimated high performance)
            'ULTIMATE_advanced_ensemble.csv': 0.47200,  # Estimated
            'ULTRA_SUPREME_ensemble.csv': 0.47180,     # Estimated
            'advanced_optimized_weights.csv': 0.47100,  # Estimated
            'ultra_geneticalgo.csv': 0.47080,          # Estimated
            'ultra_neuralmeta.csv': 0.47060,           # Estimated
            'advanced_bayesian_averaging.csv': 0.47040, # Estimated
            'ultra_top3consensus.csv': 0.47020,        # Estimated
            'advanced_stacked_meta.csv': 0.47000,      # Estimated
        }
        
        print("🌟 Loading ALL High-Performance Submissions (Original + Advanced)...")
        
        for filename, score in all_files.items():
            try:
                df = pd.read_csv(filename)
                if 'Id' in df.columns and 'Label' in df.columns:
                    self.all_submissions[filename] = df['Label'].values
                    self.scores[filename] = score
                    marker = "🆕" if "advanced" in filename or "ULTRA" in filename or "ULTIMATE" in filename else "📁"
                    print(f"{marker} {filename}: {score:.5f}")
            except Exception as e:
                print(f"❌ {filename}: Could not load")
        
        print(f"\n🚀 Loaded {len(self.all_submissions)} total submissions")
        
    def differential_evolution_optimizer(self, pred_matrix):
        """Use Differential Evolution for global weight optimization"""
        print("\n🔬 Differential Evolution Global Optimization...")
        
        n_models = pred_matrix.shape[1]
        
        def objective_function(weights):
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-8)
            
            ensemble_scores = np.dot(pred_matrix, weights)
            ensemble_pred = (ensemble_scores >= 0.5).astype(int)
            
            # Multi-objective optimization
            pos_rate = ensemble_pred.mean()
            target_rate = 0.099  # Slightly below 10% based on best performers
            
            # Primary objective: match target positive rate
            rate_penalty = abs(pos_rate - target_rate) * 100
            
            # Secondary: maximize weighted score (favor high-performing models)
            score_weights = np.array([self.scores[f] for f in self.all_submissions.keys()])
            score_weights = score_weights / score_weights.sum()
            weighted_performance = np.dot(weights, score_weights)
            
            # Tertiary: prediction diversity
            diversity_bonus = np.std(ensemble_scores) * 2
            
            # Minimize rate penalty, maximize performance and diversity
            return rate_penalty - weighted_performance - diversity_bonus
        
        # Bounds: each weight between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Run differential evolution
        result = differential_evolution(
            objective_function, 
            bounds, 
            maxiter=100,
            popsize=30,
            seed=42,
            atol=1e-6
        )
        
        optimal_weights = result.x
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        print(f"🔬 DE optimization complete. Final objective: {result.fun:.6f}")
        
        return optimal_weights
    
    def hierarchical_ensemble(self, pred_matrix):
        """Create hierarchical ensemble based on performance tiers"""
        print("\n🏗️ Creating Hierarchical Ensemble...")
        
        scores = np.array([self.scores[f] for f in self.all_submissions.keys()])
        
        # Define performance tiers
        tier1_threshold = 0.471  # Elite tier
        tier2_threshold = 0.468  # High tier  
        tier3_threshold = 0.465  # Good tier
        
        tier1_mask = scores >= tier1_threshold
        tier2_mask = (scores >= tier2_threshold) & (scores < tier1_threshold)
        tier3_mask = (scores >= tier3_threshold) & (scores < tier2_threshold)
        
        print(f"Tier 1 (Elite): {tier1_mask.sum()} models")
        print(f"Tier 2 (High): {tier2_mask.sum()} models")
        print(f"Tier 3 (Good): {tier3_mask.sum()} models")
        
        # Create tier ensembles
        tier_predictions = []
        tier_weights = []
        
        if tier1_mask.sum() > 0:
            tier1_scores = scores[tier1_mask]
            tier1_weights = tier1_scores / tier1_scores.sum()
            tier1_pred = np.dot(pred_matrix[:, tier1_mask], tier1_weights)
            tier_predictions.append(tier1_pred)
            tier_weights.append(0.6)  # Highest weight for elite tier
        
        if tier2_mask.sum() > 0:
            tier2_scores = scores[tier2_mask]
            tier2_weights = tier2_scores / tier2_scores.sum()
            tier2_pred = np.dot(pred_matrix[:, tier2_mask], tier2_weights)
            tier_predictions.append(tier2_pred)
            tier_weights.append(0.3)  # Medium weight
        
        if tier3_mask.sum() > 0:
            tier3_scores = scores[tier3_mask]
            tier3_weights = tier3_scores / tier3_scores.sum()
            tier3_pred = np.dot(pred_matrix[:, tier3_mask], tier3_weights)
            tier_predictions.append(tier3_pred)
            tier_weights.append(0.1)  # Lower weight
        
        # Combine tiers
        if tier_predictions:
            tier_weights = np.array(tier_weights)
            tier_weights = tier_weights / tier_weights.sum()
            
            hierarchical_scores = np.zeros(pred_matrix.shape[0])
            for pred, weight in zip(tier_predictions, tier_weights):
                hierarchical_scores += weight * pred
                
            hierarchical_pred = (hierarchical_scores >= 0.5).astype(int)
            
            return hierarchical_pred, hierarchical_scores
        else:
            return None, None
    
    def adaptive_confidence_ensemble(self, pred_matrix):
        """Adaptive ensemble based on prediction confidence"""
        print("\n🎯 Adaptive Confidence Ensemble...")
        
        # Calculate prediction confidence for each sample
        agreement_scores = pred_matrix.mean(axis=1)  # How many models agree
        prediction_variance = pred_matrix.var(axis=1)  # Prediction uncertainty
        
        # High confidence: high agreement, low variance
        confidence_scores = agreement_scores * (1 - prediction_variance)
        
        # Use different thresholds based on confidence
        high_conf_mask = confidence_scores >= np.percentile(confidence_scores, 80)
        med_conf_mask = (confidence_scores >= np.percentile(confidence_scores, 40)) & ~high_conf_mask
        low_conf_mask = ~(high_conf_mask | med_conf_mask)
        
        print(f"High confidence samples: {high_conf_mask.sum()}")
        print(f"Medium confidence samples: {med_conf_mask.sum()}")
        print(f"Low confidence samples: {low_conf_mask.sum()}")
        
        # Create adaptive predictions
        adaptive_pred = np.zeros(pred_matrix.shape[0], dtype=int)
        
        # High confidence: use lower threshold (more liberal)
        if high_conf_mask.sum() > 0:
            weights = np.array([self.scores[f] for f in self.all_submissions.keys()])
            weights = weights / weights.sum()
            high_conf_scores = np.dot(pred_matrix[high_conf_mask], weights)
            adaptive_pred[high_conf_mask] = (high_conf_scores >= 0.4).astype(int)
        
        # Medium confidence: standard threshold
        if med_conf_mask.sum() > 0:
            weights = np.array([self.scores[f] for f in self.all_submissions.keys()])
            weights = weights / weights.sum()
            med_conf_scores = np.dot(pred_matrix[med_conf_mask], weights)
            adaptive_pred[med_conf_mask] = (med_conf_scores >= 0.5).astype(int)
        
        # Low confidence: use higher threshold (more conservative)
        if low_conf_mask.sum() > 0:
            weights = np.array([self.scores[f] for f in self.all_submissions.keys()])
            weights = weights / weights.sum()
            low_conf_scores = np.dot(pred_matrix[low_conf_mask], weights)
            adaptive_pred[low_conf_mask] = (low_conf_scores >= 0.6).astype(int)
        
        return adaptive_pred
    
    def meta_stacking_v2(self, pred_matrix):
        """Advanced meta-stacking with multiple meta-learners"""
        print("\n🎓 Advanced Meta-Stacking v2...")
        
        # Create rich features
        n_samples, n_models = pred_matrix.shape
        
        # Feature engineering
        features = [pred_matrix]  # Base predictions
        
        # Add interaction features
        for i in range(min(5, n_models)):  # Top 5 models
            for j in range(i+1, min(5, n_models)):
                interaction = (pred_matrix[:, i] * pred_matrix[:, j]).reshape(-1, 1)
                features.append(interaction)
        
        # Statistical features
        features.append(pred_matrix.mean(axis=1).reshape(-1, 1))
        features.append(pred_matrix.std(axis=1).reshape(-1, 1))
        features.append((pred_matrix > 0.5).sum(axis=1).reshape(-1, 1))  # Count of positive predictions
        
        # Performance-weighted features
        scores = np.array([self.scores[f] for f in self.all_submissions.keys()])
        perf_weights = scores / scores.sum()
        weighted_avg = np.dot(pred_matrix, perf_weights).reshape(-1, 1)
        features.append(weighted_avg)
        
        X = np.hstack(features)
        
        # Create pseudo-labels using top 2 models
        top_indices = np.argsort(scores)[-2:]
        y = pred_matrix[:, top_indices].mean(axis=1)
        y_binary = (y >= 0.5).astype(int)
        
        # Train ensemble of meta-learners
        meta_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=8)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(30, 15), random_state=42, max_iter=500))
        ]
        
        # Cross-validation predictions
        cv_preds = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in meta_models:
            fold_preds = np.zeros(n_samples)
            
            for train_idx, val_idx in skf.split(X, y_binary):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y_binary[train_idx]
                
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    val_pred = model.predict_proba(X_val)[:, 1]
                else:
                    val_pred = model.predict(X_val).astype(float)
                
                fold_preds[val_idx] = val_pred
            
            cv_preds.append(fold_preds)
        
        # Ensemble the meta-learners
        meta_ensemble = np.mean(cv_preds, axis=0)
        meta_binary = (meta_ensemble >= 0.5).astype(int)
        
        return meta_binary, meta_ensemble
    
    def save_submission(self, predictions, filename):
        """Save submission with analysis"""
        try:
            sample_file = list(self.all_submissions.keys())[0]
            df = pd.read_csv(sample_file)
            ids = df['Id'].values
            
            submission = pd.DataFrame({
                'Id': ids,
                'Label': predictions
            })
            
            submission.to_csv(filename, index=False)
            
            pos_count = predictions.sum()
            pos_rate = pos_count / len(predictions)
            print(f"💾 {filename}: {pos_count} positives ({pos_rate:.3f} rate)")
            
        except Exception as e:
            print(f"❌ Save failed for {filename}: {e}")
    
    def run_supreme_optimization(self):
        """Run the supreme final optimization"""
        print("=" * 80)
        print("👑👑👑 SUPREME FINAL ENSEMBLE - BEAT 0.47130! 👑👑👑")
        print("=" * 80)
        
        self.load_all_submissions()
        
        if len(self.all_submissions) < 5:
            print("❌ Need at least 5 submissions!")
            return
        
        # Create prediction matrix
        pred_matrix = np.column_stack(list(self.all_submissions.values()))
        
        print(f"\n📈 Matrix shape: {pred_matrix.shape}")
        print(f"🎯 Target: Beat current best of 0.47130")
        
        final_ensembles = []
        ensemble_names = []
        
        # Method 1: Differential Evolution Optimization
        try:
            de_weights = self.differential_evolution_optimizer(pred_matrix)
            de_scores = np.dot(pred_matrix, de_weights)
            de_pred = (de_scores >= 0.5).astype(int)
            self.save_submission(de_pred, 'SUPREME_differential_evolution.csv')
            final_ensembles.append(de_pred)
            ensemble_names.append("DifferentialEvolution")
        except Exception as e:
            print(f"DE failed: {e}")
        
        # Method 2: Hierarchical Ensemble
        try:
            hier_pred, hier_scores = self.hierarchical_ensemble(pred_matrix)
            if hier_pred is not None:
                self.save_submission(hier_pred, 'SUPREME_hierarchical.csv')
                final_ensembles.append(hier_pred)
                ensemble_names.append("Hierarchical")
        except Exception as e:
            print(f"Hierarchical failed: {e}")
        
        # Method 3: Adaptive Confidence
        try:
            adaptive_pred = self.adaptive_confidence_ensemble(pred_matrix)
            self.save_submission(adaptive_pred, 'SUPREME_adaptive_confidence.csv')
            final_ensembles.append(adaptive_pred)
            ensemble_names.append("AdaptiveConfidence")
        except Exception as e:
            print(f"Adaptive confidence failed: {e}")
        
        # Method 4: Advanced Meta-Stacking
        try:
            meta_pred, meta_scores = self.meta_stacking_v2(pred_matrix)
            self.save_submission(meta_pred, 'SUPREME_meta_stacking.csv')
            final_ensembles.append(meta_pred)
            ensemble_names.append("MetaStacking")
        except Exception as e:
            print(f"Meta-stacking failed: {e}")
        
        # Method 5: Ultra-Conservative Elite Consensus
        try:
            # Only use the very best models with strict consensus
            scores = np.array([self.scores[f] for f in self.all_submissions.keys()])
            elite_threshold = 0.470
            elite_mask = scores >= elite_threshold
            
            if elite_mask.sum() >= 3:
                elite_matrix = pred_matrix[:, elite_mask]
                elite_scores = scores[elite_mask]
                elite_weights = (elite_scores ** 3) / (elite_scores ** 3).sum()  # Cube to heavily favor best
                
                elite_ensemble = np.dot(elite_matrix, elite_weights)
                # Very strict threshold to maintain quality
                elite_pred = (elite_ensemble >= 0.6).astype(int)
                
                self.save_submission(elite_pred, 'SUPREME_elite_consensus.csv')
                final_ensembles.append(elite_pred)
                ensemble_names.append("EliteConsensus")
        except Exception as e:
            print(f"Elite consensus failed: {e}")
        
        # SUPREME META-META-ENSEMBLE
        if final_ensembles:
            print("\n👑 Creating SUPREME Meta-Meta-Ensemble...")
            
            supreme_matrix = np.column_stack(final_ensembles)
            
            # Weight by target rate similarity (closer to 10% is better)
            target_rate = 0.099
            method_weights = []
            for pred in final_ensembles:
                rate = pred.mean()
                weight = 1.0 / (1.0 + abs(rate - target_rate) * 50)  # Strong penalty for deviation
                method_weights.append(weight)
            
            method_weights = np.array(method_weights)
            method_weights = method_weights / method_weights.sum()
            
            print("\nSupreme method weights:")
            for name, weight in zip(ensemble_names, method_weights):
                print(f"  {name}: {weight:.3f}")
            
            # Final supreme ensemble
            supreme_scores = np.dot(supreme_matrix, method_weights)
            supreme_pred = (supreme_scores >= 0.5).astype(int)
            
            self.save_submission(supreme_pred, 'SUPREME_META_CHAMPION.csv')
            
            print("\n" + "=" * 80)
            print("👑🏆 SUPREME OPTIMIZATION COMPLETE! 🏆👑")
            print("=" * 80)
            print("\n🥇 SUBMISSION PRIORITY ORDER:")
            print("1. 👑 SUPREME_META_CHAMPION.csv (ULTIMATE CHAMPION)")
            print("2. 🔬 SUPREME_differential_evolution.csv")
            print("3. 🏗️ SUPREME_hierarchical.csv")
            print("4. 🎯 SUPREME_adaptive_confidence.csv")
            print("5. 🎓 SUPREME_meta_stacking.csv")
            print("6. 💎 SUPREME_elite_consensus.csv")
            print("\n💪 TARGET: BEAT 0.47130 AND REACH 0.475+!")

if __name__ == "__main__":
    supreme = SupremeFinalEnsemble()
    supreme.run_supreme_optimization() 