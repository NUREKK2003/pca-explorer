import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from menu import menu
from plotly.subplots import make_subplots
from pandas.api.types import is_numeric_dtype 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

buffer = io.BytesIO()
to_export = ""


default_discrete_colors_pallete = px.colors.qualitative.Plotly
default_size = 4.00
default_alpha = 1
default_theshold = 0.7
default_variable_to_color_transformation = "none"
possible_transformations = ['none','log','standarization']
def_selected_columns = []
default_size_by_variable_name = ""
separator = ';'

st.set_page_config(layout='wide')

def download_to_xslt():
    file_path = 'export_file.xlsx'

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        st.session_state.scores.to_excel(writer, sheet_name='scores', index=True)
        st.session_state.all_loadings.to_excel(writer, sheet_name='loadings', index=True)
        if st.session_state.color_by_variable != "":
            st.session_state.copied_df[st.session_state.color_by_variable].to_excel(writer, sheet_name='vector_with_dependent_variable', index=False)
    return file_path
        
def export_and_download_to_xslt():
        path = download_to_xslt()
        with open(path, 'rb') as f:
            if st.download_button("Download to XSLT",data=f,file_name='large_df.xlsx'):
                st.success("Downloaded file!")
def loadings_plot(loadings, n_pcs=2,start_index=0,treshold=0.7):
    loadings = loadings.iloc[:,start_index:n_pcs+start_index]
    col=[]
    for x in range(0,len(loadings.columns)):
        for y in loadings.iloc[:,x]:
            if (y >= treshold) or (y <= -treshold):
                col.append('#35B2F0')
            else:
                col.append('#C2CCD0')
    col2 = np.asarray(col)
    col2 = col2.reshape(loadings.shape[1], loadings.shape[0])

    fig = make_subplots(
        rows=1, cols=len(loadings.columns),
        shared_yaxes=True,
        subplot_titles=['PC' + str(i+1+start_index) + ' - Normalized loadings' for i in range(len(loadings.columns))],
        print_grid=True
    )

    for x in range(len(loadings.columns)):
        fig.add_trace(
            go.Bar(
                y=loadings.index,
                x=loadings.iloc[:, x],
                orientation='h',
                marker_color=col2[x],
                showlegend=False
            ),
            row=1, col=x+1
        )

        fig.add_shape(
            type="line",
            x0=-treshold,
            y0=-0.5,
            x1=-treshold,
            y1=len(loadings.index)-0.5,
            line=dict(color="gray", dash="dash"),
            row=1, col=x+1
        )

        fig.add_shape(
            type="line",
            x0=treshold,
            y0=-0.5,
            x1=treshold,
            y1=len(loadings.index)-0.5,
            line=dict(color="gray", dash="dash"),
            row=1, col=x+1
        )

        fig.add_shape(
            type="line",
            x0=-1,
            y0=-0.5,
            x1=-1,
            y1=len(loadings.index)-0.5,
            line=dict(color="gray"),
            row=1, col=x+1
        )

        fig.add_shape(
            type="line",
            x0=1,
            y0=-0.5,
            x1=1,
            y1=len(loadings.index)-0.5,
            line=dict(color="gray"),
            row=1, col=x+1
        )
        for i in range(0,8):
            fig.add_shape(
                type="line",
                x0=-0.75+0.25*i,
                y0=-0.5,
                x1=-0.75+0.25*i,
                y1=len(loadings.index)-0.5,
                line=dict(color="gray",width=0.5),
                row=1, 
                col=x+1,
                opacity=0.5
            )


    fig.update_layout(
        xaxis=dict(tickvals=[-1,-0.75,-0.5,-0.25, -treshold, 0, treshold,0.25,0.5,0.75, 1], title='Loadings'),
        yaxis=dict(title='Features'),
    )
    st.plotly_chart(fig,on_select="rerun")

def create_cumulative_explained_variance_plot(explained_df):
    explained_df['cummulative_variance'] = explained_df['explained_variance'].cumsum()
    fig = px.bar(
        explained_df,
        y='explained_variance',
        x='components',
        title="Cumulative bar plot - Component vs Explainded variance")
    
    fig.add_trace(
        go.Scatter(
            x=explained_df['components'],
            y = explained_df['cummulative_variance'],
            mode='lines+markers',
            name='Explainded variance cumsum',
            line=dict(color="red")
        )
    )
    
    event = st.plotly_chart(fig,on_select="rerun")

def convert_to_nums(df):
        cats = []
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                pass
            else:
                cats.append(col)
        for col in cats:
            df[col] = pd.Categorical(df[col]).codes
        return df

st.title("Data analytics")
menu()

def create_main_scatter_plot(pcs_names,principal_df,explained_ratio,x_ax,y_ay):
    principal_df.columns = pcs_names

    if st.session_state.coloring_mode != "auto":
        fig = px.scatter(
            principal_df,
            x=x_ax,
            y=y_ay,
            title=f"Scatter plot {x_ax}({round(explained_ratio[pcs_names.index(x_ax)]*100,2)}%) vs {y_ay}({round(explained_ratio[pcs_names.index(y_ay)]*100,2)}%)",
            color=st.session_state.data_for_coloring,
            color_continuous_scale=st.session_state.choosen_pallete,
            color_discrete_sequence=st.session_state.choosen_points_color,
            size= np.asarray(st.session_state.variable_to_size_transformation)
            )
    else:
        fig = px.scatter(
            principal_df,
            x=x_ax,
            y=y_ay,
            title=f"Scatter plot {x_ax}({round(explained_ratio[pcs_names.index(x_ax)]*100,2)}%) vs {y_ay}({round(explained_ratio[pcs_names.index(y_ay)]*100,2)}%)",
            color_discrete_sequence=st.session_state.choosen_points_color,
            size= np.asarray(st.session_state.variable_to_size_transformation)
            )
    if st.session_state.size_by_variable_name == "":
         fig.update_traces(marker=dict(size=st.session_state.choosen_dot_size))
        
    event = st.plotly_chart(fig,on_select="rerun")


def pca_tab_fun(normalized_df):
    orig_df = st.session_state.copied_df


    X = normalized_df.select_dtypes(include=np.number)
    features = normalized_df.select_dtypes(include=np.number).columns.values
    x = normalized_df.loc[:,features].values

    normalized_df = pd.DataFrame(x,columns=features)
    pca = PCA()

    principal_components_df = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components_df)
    explained_ratio = pca.explained_variance_ratio_
    

    pcs_names = ['PC_' + str(x+1) for x in range(principal_df.shape[1])]

    explained_df = pd.DataFrame(explained_ratio)
    explained_df['components'] = pcs_names
    explained_df.columns.values[0] = "explained_variance"
    create_cumulative_explained_variance_plot(explained_df=explained_df)
    x_ax = st.selectbox(label="Choose PC for the x axis",options=pcs_names,index=0)
    y_ay = st.selectbox(label="Choose PC for the y axis",options=pcs_names,index=1)

    # preparing second graph
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), index=X.index.values,columns=X.columns.values)
    pca.fit(Z)
    scores = pd.DataFrame(pca.transform(Z),columns=pcs_names,index=Z.index.values)
    st.session_state.scores = scores
    Z_with_scores = Z.join(scores)
    Z_with_scores_corr = Z_with_scores.corr()
    all_loadings = (Z_with_scores_corr.iloc[len(Z.columns):,:len(pcs_names)]).T
    st.session_state.all_loadings = all_loadings




    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Plots")
        create_main_scatter_plot(pcs_names,principal_df,explained_ratio,x_ax,y_ay) 
        loadings_plot(all_loadings,1,pcs_names.index(x_ax),treshold=st.session_state.theshold)
        loadings_plot(all_loadings,1,pcs_names.index(y_ay),treshold=st.session_state.theshold)
    with col2:
        st.subheader("Markers color customization")
        choosen_pallete = px.colors.sequential.Plasma
        choosen_points_color = default_discrete_colors_pallete


        color_by_variable_check = st.radio("Set colors mode",['Automatic','By variable'],horizontal=True)
        selected_color = ""
        variable_to_color_transformation = ""

        if color_by_variable_check == 'By variable':
            st.session_state.coloring_mode = 'by-var'
            selected_color = st.selectbox("Set variable to color",options=orig_df.columns,index=0)
            variable_to_color_transformation = st.selectbox("Set variable transformation",options=possible_transformations,index=0)
            choosen_pallete = st.selectbox("Set continous color pallete",options=px.colors.named_colorscales(),index=0)
            choosen_points_color =  st.multiselect("Set discrete color pallete",options=["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure",
                "beige", "bisque", "black", "blanchedalmond", "blue",
                "blueviolet", "brown", "burlywood", "cadetblue",
                "chartreuse", "chocolate", "coral", "cornflowerblue",
                "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
                "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
                "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange",
                "darkorchid", "darkred", "darksalmon", "darkseagreen",
                "darkslateblue", "darkslategray", "darkslategrey",
                "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
                "dimgray", "dimgrey", "dodgerblue", "firebrick",
                "floralwhite", "forestgreen", "fuchsia", "gainsboro",
                "ghostwhite", "gold", "goldenrod", "gray", "grey", "green",
                "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
                "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
                "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
                "lightgoldenrodyellow", "lightgray", "lightgrey",
                "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
                "lightskyblue", "lightslategray", "lightslategrey",
                "lightsteelblue", "lightyellow", "lime", "limegreen",
                "linen", "magenta", "maroon", "mediumaquamarine",
                "mediumblue", "mediumorchid", "mediumpurple",
                "mediumseagreen", "mediumslateblue", "mediumspringgreen",
                "mediumturquoise", "mediumvioletred", "midnightblue",
                "mintcream", "mistyrose", "moccasin", "navajowhite", "navy",
                "oldlace", "olive", "olivedrab", "orange", "orangered",
                "orchid", "palegoldenrod", "palegreen", "paleturquoise",
                "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
                "plum", "powderblue", "purple", "red", "rosybrown",
                "royalblue", "saddlebrown", "salmon", "sandybrown",
                "seagreen", "seashell", "sienna", "silver", "skyblue",
                "slateblue", "slategray", "slategrey", "snow", "springgreen",
                "steelblue", "tan", "teal", "thistle", "tomato", "turquoise",
                "violet", "wheat", "white", "whitesmoke", "yellow",
                "yellowgreen"],default=['red',"maroon","lavender","dodgerblue"])
        else:
            st.session_state.coloring_mode = 'auto'

        st.subheader("Markers size customization")
        size_by_variable_check = st.radio("Set size mode",['Automatic','By variable'],horizontal=True)
        size_by_variable_name = ""
        if size_by_variable_check == 'By variable':
            st.session_state.size_mode = 'by-var'
            size_by_variable_name = st.selectbox("Set variable to adjust size",options=orig_df.columns,index=0)
        else:
            st.session_state.size_mode = 'auto'

        choosen_dot_size =st.slider("Choose dots size",0.01,10.00,default_size,0.10)

        st.subheader("Set treshold value for loadings")
        treshold_val = st.slider(label="Set threshold value", min_value=0.00, step=0.01, max_value=1.00, value=0.70, format="%f",)

        if (st.session_state.color_by_variable != selected_color) or st.session_state.choosen_pallete != choosen_pallete or choosen_points_color != st.session_state.choosen_points_color or  (choosen_dot_size != st.session_state.choosen_dot_size) or (size_by_variable_name != st.session_state.size_by_variable_name) or (treshold_val != st.session_state.theshold) or (variable_to_color_transformation != st.session_state.variable_to_color_transformation):
            st.session_state.color_by_variable = selected_color
            st.session_state.choosen_pallete = choosen_pallete
            st.session_state.choosen_points_color = choosen_points_color
            st.session_state.choosen_dot_size = choosen_dot_size
            st.session_state.size_by_variable_name = size_by_variable_name
            st.session_state.choosen_alpha = default_alpha
            st.session_state.theshold = treshold_val
            st.session_state.variable_to_color_transformation = variable_to_color_transformation
            if st.session_state.coloring_mode == 'by-var':
                if(variable_to_color_transformation == default_variable_to_color_transformation):
                    st.session_state.data_for_coloring = orig_df[selected_color]
                elif(variable_to_color_transformation == 'log'):
                    st.session_state.data_for_coloring = np.log(orig_df[selected_color])
                elif(variable_to_color_transformation == 'standarization'):
                    st.session_state.data_for_coloring = ((orig_df[selected_color] - orig_df[selected_color].mean())/orig_df[selected_color].std()).fillna(0)
                print(st.session_state.data_for_coloring.head())
                print(selected_color)
            if st.session_state.size_mode =='by-var':
                if size_by_variable_name != "":
                    val_standarized = ((orig_df[size_by_variable_name] - orig_df[size_by_variable_name].mean())/orig_df[size_by_variable_name].std()).fillna(0)
                    val_min = min(val_standarized)
                    st.session_state.variable_to_size_transformation = val_standarized + abs(val_min)
            st.rerun()

        if st.button("Generate XSLT"):
            path = download_to_xslt()
            with open(path, 'rb') as f:
                if st.download_button("Download to XSLT",data=f,file_name='large_df.xlsx'):
                    st.success("Downloaded file!")

prep_tab,pca_tab = st.tabs(['I. Data preparation','II. PCA'])

with prep_tab:
    @st.cache_data(persist="disk")
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file, sep=separator)
        df = df.dropna()
        
        df_not_number = df.select_dtypes(exclude="number")
        df_not_number = df_not_number.astype("category")
        for column in df_not_number.columns:
            df_not_number[column] = df_not_number[column].cat.codes
        
        df_numeric = df.select_dtypes(include="number")
        
        df = pd.concat([df_numeric,df_not_number],axis=1)
        return df

    separator = st.text_input("Enter document separator:", value=";")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    df_data = None
    
    if uploaded_file is not None:
        try:
            df_data = load_data(uploaded_file)
            st.write("Here's a preview of your data:")
            st.dataframe(df_data.head())
            default_size_by_variable_name = df_data.columns[-1]
        except Exception as e:
            st.error(f"Error processing the CSV file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

    if df_data is not None:

        st.write('Loaded '+str(df_data.shape[0])+' rows')
        
        st.session_state.selected_dataset = "all data"
        with st.form("observations_filter_form"):
            st.write("Select filtering parameters")

            def_selected_columns = df_data.columns.values.tolist()
            multiselect = st.multiselect("Select the collumns names that you want to use in PCA analyse",df_data.columns.values.tolist(),default=def_selected_columns)

            submitted = st.form_submit_button("Submit")
            if submitted:
                if len(list(multiselect))<2:
                    st.error('Select at least 2 columns!', icon="🚨")
                else:
                    st.session_state.pca_unlocked = 1
                    st.session_state.selected_columns = multiselect


                    st.session_state.copied_df = df_data.copy(deep=True)

                    df_data = df_data[multiselect]

                    # mapping no numbers columns  to numbers
                    df_data = convert_to_nums(df_data)

                    # normalization/standarization

                    df_data=(df_data-df_data.mean())/df_data.std()
                    df_data = df_data.fillna(0)


                    st.session_state.coloring_mode = 'auto'
                    st.session_state.size_mode = 'auto'

                    st.session_state.prepared_df = df_data
                    st.session_state.color_by_variable = ""
                    st.session_state.choosen_pallete = px.colors.sequential.Plasma
                    st.session_state.choosen_points_color = default_discrete_colors_pallete
                    st.session_state.choosen_dot_size = default_size
                    st.session_state.choosen_alpha = default_alpha
                    st.session_state.size_by_variable_name = ""
                    st.session_state.theshold = default_theshold
                    st.session_state.variable_to_color_transformation = ""
                    val_standarized = ((st.session_state.copied_df[default_size_by_variable_name] - st.session_state.copied_df[default_size_by_variable_name].mean())/st.session_state.copied_df[default_size_by_variable_name].std()).fillna(0)
                    st.session_state.variable_to_size_transformation = val_standarized + abs(min(val_standarized))
                    st.success('Data is ready! Now you can go to PCA page!', icon="✅")
                    print(df_data.shape)
                

with pca_tab:
    if 'pca_unlocked' in st.session_state and st.session_state.pca_unlocked == 1:
        st.write("PCA analyse")
        pca_tab_fun(st.session_state.prepared_df)
    else:
        st.error("Please select the data first", icon="🚨")