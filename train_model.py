"""
AcademIA - Sistema de Previsão de Desempenho Escolar
Inteligência Artificial aplicada à Educação
Análise, Treinamento e Avaliação do Modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Configuração de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Criar diretórios se não existirem
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('images', exist_ok=True)
print("✓ Diretórios criados/verificados")

# ============================================================================
# 1. GERAÇÃO DE DATASET SINTÉTICO (ou use um dataset real)
# ============================================================================

def create_student_dataset(n_samples=1000):
    """Cria dataset sintético de desempenho escolar"""
    np.random.seed(42)
    
    data = {
        'horas_estudo_semanal': np.random.randint(0, 40, n_samples),
        'frequencia_aulas': np.random.uniform(60, 100, n_samples),
        'nota_anterior': np.random.uniform(3, 10, n_samples),
        'participacao_atividades': np.random.randint(0, 20, n_samples),
        'horas_sono': np.random.uniform(4, 10, n_samples),
        'nivel_estresse': np.random.randint(1, 11, n_samples),
        'apoio_familiar': np.random.randint(1, 6, n_samples),
        'acesso_internet': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'turno': np.random.choice(['Manhã', 'Tarde', 'Noite'], n_samples),
        'transporte_tempo': np.random.randint(0, 120, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Criar nota final baseada em fatores realistas
    df['nota_final'] = (
        df['horas_estudo_semanal'] * 0.12 +
        df['frequencia_aulas'] * 0.05 +
        df['nota_anterior'] * 0.4 +
        df['participacao_atividades'] * 0.15 +
        df['horas_sono'] * 0.3 -
        df['nivel_estresse'] * 0.2 +
        df['apoio_familiar'] * 0.3 +
        df['acesso_internet'] * 1.5 -
        df['transporte_tempo'] * 0.01 +
        np.random.normal(0, 1, n_samples)
    )
    
    # Normalizar nota final entre 0 e 10
    df['nota_final'] = df['nota_final'].clip(0, 10)
    
    return df

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA
# ============================================================================

def exploratory_analysis(df):
    """Realiza análise exploratória dos dados"""
    print("=" * 80)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 80)
    
    print("\n1. Informações Gerais:")
    print(df.info())
    
    print("\n2. Estatísticas Descritivas:")
    print(df.describe())
    
    print("\n3. Valores Nulos:")
    print(df.isnull().sum())
    
    # Correlação
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Matriz de correlação salva em 'images/correlation_matrix.png'")
    
    # Distribuição da nota final
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df['nota_final'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Nota Final')
    plt.ylabel('Frequência')
    plt.title('Distribuição da Nota Final')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['nota_final'])
    plt.ylabel('Nota Final')
    plt.title('Boxplot da Nota Final')
    plt.tight_layout()
    plt.savefig('images/target_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Distribuição da variável alvo salva em 'images/target_distribution.png'")

# ============================================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================================

def preprocess_data(df):
    """Prepara os dados para treinamento"""
    df_processed = df.copy()
    
    # Encoding de variável categórica
    le = LabelEncoder()
    df_processed['turno_encoded'] = le.fit_transform(df_processed['turno'])
    
    # Salvar encoder
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Separar features e target
    X = df_processed.drop(['nota_final', 'turno'], axis=1)
    y = df_processed['nota_final']
    
    return X, y, le

# ============================================================================
# 4. TREINAMENTO E AVALIAÇÃO
# ============================================================================

def train_and_evaluate_models(X, y):
    """Treina e avalia múltiplos modelos"""
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Salvar scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Modelos a testar
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Treinamento
        model.fit(X_train_scaled, y_train)
        
        # Predições
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métricas
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=5, scoring='r2')
        
        print(f"R² (Treino): {train_r2:.4f}")
        print(f"R² (Teste): {test_r2:.4f}")
        print(f"RMSE (Teste): {test_rmse:.4f}")
        print(f"MAE (Teste): {test_mae:.4f}")
        print(f"Cross-validation R² (média): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test
        }
    
    # Selecionar melhor modelo
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'=' * 80}")
    print(f"MELHOR MODELO: {best_model_name}")
    print(f"{'=' * 80}")
    
    # Salvar melhor modelo
    joblib.dump(best_model, 'models/best_model.pkl')
    print("✓ Modelo salvo em 'models/best_model.pkl'")
    
    # Visualizar predições
    visualize_predictions(y_test, results[best_model_name]['predictions'])
    
    # Feature importance (se disponível)
    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, X.columns)
    
    return best_model, scaler, X_test, y_test, results

# ============================================================================
# 5. VISUALIZAÇÕES
# ============================================================================

def visualize_predictions(y_true, y_pred):
    """Visualiza predições vs valores reais"""
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Nota Real')
    plt.ylabel('Nota Predita')
    plt.title('Predições vs Valores Reais')
    
    # Resíduos
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Nota Predita')
    plt.ylabel('Resíduos')
    plt.title('Análise de Resíduos')
    
    plt.tight_layout()
    plt.savefig('images/predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Análise de predições salva em 'images/predictions_analysis.png'")

def plot_feature_importance(model, feature_names):
    """Plota importância das features"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importância')
    plt.title('Importância das Variáveis')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Importância das features salva em 'images/feature_importance.png'")

# ============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Criar dataset
    print("Criando dataset...")
    df = create_student_dataset(n_samples=1000)
    df.to_csv('data/student_performance_dataset.csv', index=False)
    print("✓ Dataset salvo em 'data/student_performance_dataset.csv'")
    
    # Análise exploratória
    exploratory_analysis(df)
    
    # Pré-processamento
    print("\nPré-processando dados...")
    X, y, label_encoder = preprocess_data(df)
    
    # Treinamento
    best_model, scaler, X_test, y_test, results = train_and_evaluate_models(X, y)
    
    print("\n" + "=" * 80)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print("\nArquivos gerados:")
    print("  📁 data/")
    print("    • student_performance_dataset.csv")
    print("  📁 models/")
    print("    • best_model.pkl")
    print("    • scaler.pkl")
    print("    • label_encoder.pkl")
    print("  📁 images/")
    print("    • correlation_matrix.png")
    print("    • target_distribution.png")
    print("    • predictions_analysis.png")
    print("    • feature_importance.png")