import streamlit as st
import streamlit.components.v1 as components
import requests
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd
import joblib
# import shap
# from fastapi.encoders import jsonable_encoder
import plotly.graph_objects as go
import pickle



########################################################
# Session for the API
########################################################
def fetch(session, url):

    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

session = requests.Session()


########################################################
# Functions to call the EndPoints
########################################################
@st.cache
def client():
    # Getting clients Id
    #response = fetch(session, f"http://fastapi:8008/api/clients")
    response = requests.get("https://api-oc-p7-mbd.herokuapp.com/api/clients").json()
    if response:
        return response["clientsId"]
    else:
        return "Error"

def client_details(id):
    # Getting client's details
    response = fetch(session, f"https://api-oc-p7-mbd.herokuapp.com/api/clients/{id}")
    if response:
        return response
    else:
        return "Error"

def client_prediction(id):
    # Getting client's prediction
    response = fetch(session, f"https://api-oc-p7-mbd.herokuapp.com/api/predictions/clients/{id}")
    if response:
        return response
    else:
        return "Error"

def clients_df(id):
    # Getting client's df based on id
    response = fetch(session, f"https://api-oc-p7-mbd.herokuapp.com/api/predictions/clients/shap/{id}")
    if response:
        return response
    else:
        return "Error"

def load_model():
    response = fetch(session, f"https://api-oc-p7-mbd.herokuapp.com/api/model/{id}")
    if response:
        return response
    else:
        return "Error"


data_test = pd.read_csv('Data/data_test.csv', index_col='SK_ID_CURR', encoding='utf-8')  #
X_test = pd.read_csv('./Data/reduced_X_test.csv', index_col='SK_ID_CURR', encoding='utf-8')

@st.cache
def load_kmeans(datadf, idclient, mdl):
    index = datadf[datadf.index == int(idclient)].index.values
    index = index[0]
    data_client = pd.DataFrame(datadf.loc[datadf.index, :])
    df_neighbors = pd.DataFrame(mdl.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data_test.drop(['Unnamed: 0'],axis=1)], axis=1)
    return df_neighbors.iloc[:,1:].sample(5)
@st.cache(allow_output_mutation=True)
def clusters():
    # pickle_in = open('Data/clustering','rb')
    # model_cluster = pickle.load(pickle_in)
    pickle_in = "./Data/clustering.joblib"
    with open(pickle_in, 'rb') as fo:
        model_cluster = joblib.load(fo)
    return (model_cluster)
########################################################
# To show the SHAP image
########################################################
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

icon = Image.open("images/favicon.ico")
image = Image.open("images/pret-a-depenser.png")
image_oc=Image.open("images/Logo_OpenClassrooms.png")
########################################################
# General settings
########################################################
st.set_page_config(
    page_title="Prêt à dépenser - Default Risk",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",

)
sb = st.sidebar
sb.image(image)
sb.markdown("### Choix du client")
client_id = sb.selectbox("id client : " , client())
df = pd.DataFrame(data=client_details(client_id))
sb.markdown("Filters")
selection_modes = ["none", "all", "other"]
selection_mode = sb.radio("", selection_modes, index=0)



st.title("Prêt à dépenser")
st.header("Projet 7 - Implémentez un modèle de scoring")
st.subheader("Openclassroom - Formation Data scientist")

# st.markdown("### 1. Client")
# st.markdown("## 1.1. Choix")
# # client_id = st.selectbox("sélection : " , client())
st.markdown("# 1. Données")
selected_columns = []
if selection_mode == "none":
    for col in df.columns:
        a = sb.checkbox(label=col, value=False, disabled=True)
    selected_columns=[]
if selection_mode == "all":
    for col in df.columns:
        # labeled_text=col + ": " + str(df[col].values)
        a = sb.checkbox(label=col, value=True, disabled=True)
        # sb.markdown(col + ": " + str(df[col].values))
    selected_columns = df.columns
if selection_mode == "other":
    for col in df.columns:
        a=sb.checkbox(label=col)
        if a==True:
            selected_columns.append(col)
        if col in selected_columns:
            if a==False:
                selected_columns.remove(col)
if len(selected_columns)>0:
    st.dataframe(data=df[selected_columns])#.style.highlight_max(axis=0), width=None, height=None)

st.markdown("# 2. Prêt")
st.markdown("## 2.1. Accord")

prediction_client=client_prediction(client_id)
#st.write(prediction_client)
# client_prediction(client_id)
if prediction_client.get("repay")=="Yes":
    st.success("Le crédit du client " + str(int(client_id)) +" est accordé!")
if prediction_client.get("repay")=="No":
    st.error("Le crédit du client " + str(int(client_id)) +" n'est pas accordé!")


st.markdown("## 2.2. graphique")
x0 = prediction_client.get("probability0")*100
x1 = prediction_client.get("probability1")*100
xthreshold=prediction_client.get("threshold")*100
fig_gauge_chart = go.Figure(go.Indicator(
   domain = {'x': [0, 1], 'y': [0, 1]},
    value = x1,
    mode = "gauge+number",
    title = {'text': "Probability"},
    gauge = {
        'axis': {'range': [None, 100]},
        "bar": {"color": "white"},# LawnGreen
        "bgcolor": "white",
        "steps":
        [{"range": [0, xthreshold], "color": "#27AE60"},#Green
        {"range": [xthreshold, 100], "color": "#E74C3C"}#red
        ]}))
col1, col2 = st.columns(2)

with col1:
    # st.header("A cat")
    st.plotly_chart(fig_gauge_chart, use_container_width=False, sharing="streamlit")

with col2:
    # st.header("A dog")
    st.write(prediction_client)
# Add histogram data

st.markdown("## 2.3. SHAP")

# if "explainer" not in st.session_state:
#     loaded_explainer_name = "./Data/explainer_reduced_X_test.pkl"
#     st.session_state["explainer"] = pickle.load(open(loaded_explainer_name, 'rb'))
# if "shap_values" not in st.session_state:
#     loaded_shap_values_name = "./Data/shap_values_reduced_X_test.pkl"
#     st.session_state["shap_values"] = pickle.load(open(loaded_shap_values_name, 'rb'))
#
# shap.initjs()
# shap.force_plot(st.session_state["explainer"].expected_value[0], st.session_state["shap_values"][0][0], df.iloc[0,:])
# st.pyplot(fig)
# shap.initjs()
# shap.bar_plot(st.session_state["explainer"].shap_values(df[0]), feature_names=np.array(feats), max_display=10)
# st.pyplot(fig)
# shap.initjs()
# shap.plots.force(st.session_state["shap_values"][0])
# st.pyplot(fig)
st.markdown("## 2.4. Clients avec profil similaire")
chk_voisins = st.checkbox("Clients avec un profil similaire  ?")

if chk_voisins:
    knn = clusters() #modele de clustering
    st.markdown("<u>la liste de 5 clients similaires :</u>", unsafe_allow_html=True)
    st.dataframe(load_kmeans(X_test, client_id, knn))
    st.markdown("<i>Target 1 = Client avec difficultés de paiment</i>", unsafe_allow_html=True)
# else:
#     st.markdown("<i>…</i>", unsafe_allow_html=True)
# #quelques informations générales
# st.header("**Informations du client**")
