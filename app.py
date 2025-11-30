"""
AcademIA - Sistema de Previsão de Desempenho Escolar
Inteligência Artificial aplicada à Educação
Aplicação Streamlit para Deploy
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="🎓 AcademIA - Predição de Desempenho",
    page_icon="🎓",
    layout="wide"
)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_resource
def load_models():
    """Carrega os modelos treinados"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("⚠️ Modelos não encontrados! Execute o script de treinamento primeiro.")
        st.info("Execute: `python train_model.py` no terminal")
        return None, None, None

def predict_performance(model, scaler, label_encoder, input_data):
    """Realiza a predição de desempenho"""
    # Preparar dados
    df_input = pd.DataFrame([input_data])
    
    # Encoding do turno
    df_input['turno_encoded'] = label_encoder.transform([input_data['turno']])[0]
    df_input = df_input.drop('turno', axis=1)
    
    # Escalar dados
    X_scaled = scaler.transform(df_input)
    
    # Predição
    prediction = model.predict(X_scaled)[0]
    
    # Garantir que a nota está entre 0 e 10
    prediction = np.clip(prediction, 0, 10)
    
    return prediction

def get_performance_category(nota):
    """Categoriza o desempenho do aluno"""
    if nota >= 9:
        return "Excelente", "#00C853"
    elif nota >= 7:
        return "Bom", "#64DD17"
    elif nota >= 5:
        return "Regular", "#FFC107"
    else:
        return "Necessita Atenção", "#FF5252"

def generate_recommendations(input_data, prediction):
    """Gera recomendações personalizadas"""
    recommendations = []
    
    if input_data['horas_estudo_semanal'] < 10:
        recommendations.append("Aumentar horas de estudo semanal (recomendado: 15-20h)")
    
    if input_data['frequencia_aulas'] < 80:
        recommendations.append("Melhorar frequência às aulas (meta: acima de 85%)")
    
    if input_data['horas_sono'] < 7:
        recommendations.append("Aumentar horas de sono (recomendado: 7-9h por noite)")
    
    if input_data['nivel_estresse'] > 7:
        recommendations.append("Gerenciar níveis de estresse (considere técnicas de relaxamento)")
    
    if input_data['participacao_atividades'] < 10:
        recommendations.append("Participar mais de atividades e exercícios")
    
    if input_data['transporte_tempo'] > 60:
        recommendations.append("Tempo de transporte elevado - considere estratégias de otimização")
    
    if not recommendations:
        recommendations.append("Continue mantendo seus hábitos de estudo!")
    
    return recommendations

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    # Título
    st.title("AcademIA - Sistema de Previsão de Desempenho Escolar")
    st.markdown("**Trabalho Final - Inteligência Artificial**")
    st.markdown("---")
    
    # Carregar modelos
    model, scaler, label_encoder = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar - Navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Selecione uma página:",
        ["Início", "Fazer Predição", "Análise dos Dados", "Sobre o AcademIA"]
    )
    
    # ========================================================================
    # PÁGINA HOME
    # ========================================================================
    if page == "Início":
        st.header("Bem-vindo ao Sistema AcademIA")
        
        st.write("""
        Este sistema utiliza técnicas de Machine Learning para prever o desempenho 
        escolar de alunos com base em diversos fatores comportamentais e acadêmicos.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Objetivo**\n\nPrever a nota final de alunos com base em fatores comportamentais e acadêmicos.")
        
        with col2:
            st.success("**Técnica**\n\nRegressão com Machine Learning usando algoritmos de aprendizado supervisionado.")
        
        with col3:
            st.warning("**Dataset**\n\n1000 registros de alunos com 10 variáveis preditoras.")
        
        st.markdown("---")
        
        st.subheader("Variáveis Analisadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Variáveis Acadêmicas:**
            - Horas de estudo semanal
            - Frequência às aulas (%)
            - Nota do período anterior
            - Participação em atividades
            - Acesso à internet
            """)
        
        with col2:
            st.markdown("""
            **Variáveis Pessoais:**
            - Horas de sono por noite
            - Nível de estresse (1-10)
            - Apoio familiar (1-5)
            - Turno de estudo
            - Tempo de transporte (minutos)
            """)
        
        st.markdown("---")
        st.info("Use o menu lateral para navegar entre as funcionalidades do sistema.")
    
    # ========================================================================
    # PÁGINA DE PREDIÇÃO
    # ========================================================================
    elif page == "Fazer Predição":
        st.header("Predição de Desempenho Escolar")
        st.write("Preencha os dados do aluno abaixo para obter a previsão de nota final.")
        
        # Formulário de entrada
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dados Acadêmicos")
                horas_estudo = st.slider(
                    "Horas de estudo por semana",
                    min_value=0, max_value=40, value=15
                )
                frequencia = st.slider(
                    "Frequência às aulas (%)",
                    min_value=60.0, max_value=100.0, value=85.0, step=0.5
                )
                nota_anterior = st.slider(
                    "Nota do período anterior",
                    min_value=0.0, max_value=10.0, value=7.0, step=0.1
                )
                participacao = st.slider(
                    "Participação em atividades (qtd)",
                    min_value=0, max_value=20, value=10
                )
                acesso_internet = st.selectbox(
                    "Possui acesso à internet?",
                    options=[1, 0],
                    format_func=lambda x: "Sim" if x == 1 else "Não"
                )
            
            with col2:
                st.subheader("Dados Pessoais")
                horas_sono = st.slider(
                    "Horas de sono por noite",
                    min_value=4.0, max_value=10.0, value=7.5, step=0.5
                )
                nivel_estresse = st.slider(
                    "Nível de estresse (1-10)",
                    min_value=1, max_value=10, value=5
                )
                apoio_familiar = st.slider(
                    "Apoio familiar (1-5)",
                    min_value=1, max_value=5, value=3
                )
                turno = st.selectbox(
                    "Turno de estudo",
                    options=["Manhã", "Tarde", "Noite"]
                )
                transporte_tempo = st.slider(
                    "Tempo de transporte (min)",
                    min_value=0, max_value=120, value=30
                )
            
            submitted = st.form_submit_button("Fazer Predição", use_container_width=True)
        
        if submitted:
            # Preparar dados de entrada
            input_data = {
                'horas_estudo_semanal': horas_estudo,
                'frequencia_aulas': frequencia,
                'nota_anterior': nota_anterior,
                'participacao_atividades': participacao,
                'horas_sono': horas_sono,
                'nivel_estresse': nivel_estresse,
                'apoio_familiar': apoio_familiar,
                'acesso_internet': acesso_internet,
                'turno': turno,
                'transporte_tempo': transporte_tempo
            }
            
            # Fazer predição
            prediction = predict_performance(model, scaler, label_encoder, input_data)
            category, color = get_performance_category(prediction)
            
            st.markdown("---")
            st.subheader("Resultado da Predição")
            
            # Exibir resultado
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 30px; 
                            background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
                            border-radius: 15px; border: 2px solid {color};'>
                    <h2 style='color: {color}; margin: 10px 0;'>{prediction:.2f}</h2>
                    <p style='font-size: 20px; margin: 0;'>{category}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Gráfico de gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Nota Prevista"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 5], 'color': "#FFCDD2"},
                        {'range': [5, 7], 'color': "#FFF9C4"},
                        {'range': [7, 9], 'color': "#C8E6C9"},
                        {'range': [9, 10], 'color': "#A5D6A7"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendações
            st.subheader("Recomendações Personalizadas")
            recommendations = generate_recommendations(input_data, prediction)
            
            for rec in recommendations:
                st.success(rec)
            
            # Análise de fatores
            st.subheader("Análise dos Fatores")
            
            factors_df = pd.DataFrame({
                'Fator': ['Horas Estudo', 'Frequência', 'Nota Anterior', 
                         'Participação', 'Horas Sono', 'Estresse', 
                         'Apoio Familiar', 'Internet', 'Transporte'],
                'Valor': [horas_estudo/40*10, frequencia/10, nota_anterior,
                         participacao/2, horas_sono, 10-nivel_estresse,
                         apoio_familiar*2, acesso_internet*10, 
                         (120-transporte_tempo)/12]
            })
            
            fig = px.bar(factors_df, x='Valor', y='Fator', orientation='h',
                        color='Valor', color_continuous_scale='RdYlGn')
            fig.update_layout(
                title="Pontuação Normalizada dos Fatores (0-10)",
                xaxis_title="Pontuação",
                yaxis_title="",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PÁGINA DE ANÁLISE EXPLORATÓRIA
    # ========================================================================
    elif page == "Análise dos Dados":
        st.header("Análise Exploratória dos Dados")
        
        # Carregar dataset
        try:
            df = pd.read_csv('data/student_performance_dataset.csv')
            
            # Estatísticas gerais
            st.subheader("Estatísticas Gerais")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Alunos", len(df))
            with col2:
                st.metric("Nota Média", f"{df['nota_final'].mean():.2f}")
            with col3:
                st.metric("Nota Máxima", f"{df['nota_final'].max():.2f}")
            with col4:
                st.metric("Nota Mínima", f"{df['nota_final'].min():.2f}")
            
            st.markdown("---")
            
            # Distribuição da nota final
            st.subheader("Distribuição da Nota Final")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='nota_final', nbins=30,
                                  title='Histograma da Nota Final')
                fig.update_layout(xaxis_title="Nota Final", yaxis_title="Frequência")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y='nota_final', 
                            title='Boxplot da Nota Final')
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlações
            st.subheader("Análise de Correlações")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto='.2f',
                           aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Matriz de Correlação')
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise por turno
            st.subheader("Análise por Turno")
            
            turno_stats = df.groupby('turno')['nota_final'].agg(['mean', 'count']).reset_index()
            
            fig = px.bar(turno_stats, x='turno', y='mean', 
                        text='mean',
                        title='Nota Média por Turno',
                        color='mean',
                        color_continuous_scale='Viridis')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Turno", yaxis_title="Nota Média")
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.error("⚠️ Dataset não encontrado! Execute o script de treinamento primeiro.")
    
    # ========================================================================
    # PÁGINA SOBRE O PROJETO
    # ========================================================================
    elif page == "Sobre o AcademIA":
        st.header("Sobre o Projeto AcademIA")
        
        st.markdown("""
        ### Objetivo do Projeto
        
        Este sistema foi desenvolvido para prever o desempenho escolar de alunos com base em 
        diversos fatores comportamentais, acadêmicos e pessoais. O objetivo é identificar 
        alunos em risco e fornecer recomendações personalizadas.
        
        ### Técnicas Utilizadas
        
        **Algoritmos de Regressão:**
        - Regressão Linear
        - Random Forest Regressor
        - Gradient Boosting Regressor
        
        **Pré-processamento:**
        - Normalização de features (StandardScaler)
        - Encoding de variáveis categóricas (LabelEncoder)
        - Validação cruzada (5-fold)
        
        ### Métricas de Desempenho
        
        O modelo é avaliado utilizando:
        - **R² Score**: Coeficiente de determinação
        - **RMSE**: Erro quadrático médio
        - **MAE**: Erro absoluto médio
        - **Cross-validation**: Validação cruzada para robustez
        
        ### Variáveis Preditoras
        
        1. **Horas de estudo semanal**: Tempo dedicado aos estudos
        2. **Frequência às aulas**: Percentual de presença
        3. **Nota anterior**: Desempenho no período anterior
        4. **Participação em atividades**: Engajamento acadêmico
        5. **Horas de sono**: Qualidade do descanso
        6. **Nível de estresse**: Saúde mental
        7. **Apoio familiar**: Suporte em casa
        8. **Acesso à internet**: Recursos tecnológicos
        9. **Turno de estudo**: Período das aulas
        10. **Tempo de transporte**: Logística diária
        
        ### Limitações
        
        **Limitações:**
        - Dataset sintético (necessita validação com dados reais)
        - Não considera alguns fatores socioeconômicos detalhados
        
        ### Referências
        
        - Scikit-learn Documentation
        - Streamlit Documentation
        """)


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    main()