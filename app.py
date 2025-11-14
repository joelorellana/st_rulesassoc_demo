"""
Sistema de Recomendación - Reglas de Asociación
Streamlit App
"""

import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Recomendación",
    page_icon="🛒",
    layout="wide"
)

# Título
st.title("🛒 Sistema de Recomendación con Apriori")
st.markdown("**Basado en Reglas de Asociación - Online Retail Dataset**")

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        rules = joblib.load('association_rules.pkl')
        return rules
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'association_rules.pkl'")
        st.info("Ejecuta el notebook de entrenamiento primero para generar el modelo.")
        return None

rules = load_model()

if rules is not None:
    # Sidebar con información
    st.sidebar.header("📊 Información del Modelo")
    st.sidebar.metric("Total de Reglas", len(rules))
    st.sidebar.metric("Support Promedio", f"{rules['support'].mean():.4f}")
    st.sidebar.metric("Confidence Promedio", f"{rules['confidence'].mean():.4f}")
    st.sidebar.metric("Lift Promedio", f"{rules['lift'].mean():.2f}")
    
    # Filtros
    st.sidebar.header("⚙️ Filtros")
    min_confidence = st.sidebar.slider("Confidence Mínimo", 0.0, 1.0, 0.3, 0.05)
    min_lift = st.sidebar.slider("Lift Mínimo", 1.0, 15.0, 1.2, 0.1)
    
    # Filtrar reglas
    rules_filtered = rules[
        (rules['confidence'] >= min_confidence) & 
        (rules['lift'] >= min_lift)
    ].copy()
    
    st.sidebar.success(f"✅ Reglas filtradas: {len(rules_filtered)}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Recomendador", "📋 Reglas", "📊 Estadísticas"])
    
    # TAB 1: RECOMENDADOR
    with tab1:
        st.header("Sistema de Recomendación")
        
        # Obtener lista de productos únicos
        all_products = set()
        for itemset in rules['antecedents']:
            all_products.update(itemset)
        all_products = sorted(list(all_products))
        
        # Selector de producto
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_product = st.selectbox(
                "Selecciona el producto que el cliente compró:",
                all_products,
                index=0
            )
        
        with col2:
            top_n = st.number_input("Top N recomendaciones", 1, 10, 5)
        
        if st.button("🔍 Buscar Recomendaciones", type="primary"):
            # Filtrar reglas
            recommendations = rules_filtered[
                rules_filtered['antecedents'].apply(
                    lambda x: selected_product in x
                )
            ].copy()
            
            if len(recommendations) > 0:
                # Ordenar por lift
                recommendations = recommendations.nlargest(top_n, 'lift')
                
                st.success(f"✅ Se encontraron {len(recommendations)} recomendaciones")
                
                # Mostrar recomendaciones
                for idx, row in recommendations.iterrows():
                    consequent = list(row['consequents'])[0]
                    
                    col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                    
                    with col_a:
                        st.markdown(f"### 📦 {consequent}")
                    with col_b:
                        st.metric("Confidence", f"{row['confidence']:.2%}")
                    with col_c:
                        st.metric("Lift", f"{row['lift']:.2f}")
                    with col_d:
                        st.metric("Support", f"{row['support']:.3f}")
                    
                    st.divider()
            else:
                st.warning(f"⚠️ No se encontraron recomendaciones para: **{selected_product}**")
                st.info("Intenta ajustar los filtros en el sidebar o selecciona otro producto.")
    
    # TAB 2: REGLAS
    with tab2:
        st.header("Tabla de Reglas de Asociación")
        
        # Preparar datos para mostrar
        display_rules = rules_filtered.copy()
        display_rules['antecedents_str'] = display_rules['antecedents'].apply(
            lambda x: ', '.join(list(x))
        )
        display_rules['consequents_str'] = display_rules['consequents'].apply(
            lambda x: ', '.join(list(x))
        )
        
        # Seleccionar columnas
        display_df = display_rules[[
            'antecedents_str', 'consequents_str', 
            'support', 'confidence', 'lift'
        ]].copy()
        
        display_df.columns = ['Antecedente', 'Consecuente', 'Support', 'Confidence', 'Lift']
        
        # Formatear
        display_df['Support'] = display_df['Support'].apply(lambda x: f"{x:.4f}")
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.4f}")
        display_df['Lift'] = display_df['Lift'].apply(lambda x: f"{x:.2f}")
        
        # Mostrar con búsqueda
        search = st.text_input("🔍 Buscar producto en reglas:", "")
        
        if search:
            mask = (
                display_df['Antecedente'].str.contains(search, case=False) |
                display_df['Consecuente'].str.contains(search, case=False)
            )
            display_df = display_df[mask]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=500
        )
        
        # Descargar CSV
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar reglas (CSV)",
            data=csv,
            file_name="reglas_asociacion.csv",
            mime="text/csv"
        )
    
    # TAB 3: ESTADÍSTICAS
    with tab3:
        st.header("Estadísticas del Modelo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Reglas",
                len(rules_filtered),
                f"{len(rules_filtered) - len(rules)} del total"
            )
        
        with col2:
            st.metric(
                "Lift Máximo",
                f"{rules_filtered['lift'].max():.2f}"
            )
        
        with col3:
            st.metric(
                "Confidence Máximo",
                f"{rules_filtered['confidence'].max():.2%}"
            )
        
        st.divider()
        
        # Gráficos
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Distribución de Lift")
            st.bar_chart(rules_filtered['lift'].value_counts().sort_index())
        
        with col_b:
            st.subheader("Distribución de Confidence")
            st.bar_chart(rules_filtered['confidence'].value_counts().sort_index())
        
        # Top 10 reglas
        st.subheader("Top 10 Reglas por Lift")
        top_10 = rules_filtered.nlargest(10, 'lift')
        
        top_10_display = pd.DataFrame({
            'Regla': [
                f"{list(a)[0][:30]} → {list(c)[0][:30]}"
                for a, c in zip(top_10['antecedents'], top_10['consequents'])
            ],
            'Lift': top_10['lift'].values,
            'Confidence': top_10['confidence'].values
        })
        
        st.dataframe(top_10_display, use_container_width=True)

else:
    st.error("No se pudo cargar el modelo. Asegúrate de tener 'association_rules.pkl' en el mismo directorio.")
    
    st.markdown("""
    ### Pasos para generar el modelo:
    
    1. Ejecuta el notebook de entrenamiento
    2. El notebook generará `association_rules.pkl`
    3. Coloca el archivo en la misma carpeta que este `app.py`
    4. Ejecuta: `streamlit run app.py`
    """)

# Footer
st.markdown("---")
st.markdown("**Desarrollado con Streamlit** | Dataset: Online Retail (UCI)")
