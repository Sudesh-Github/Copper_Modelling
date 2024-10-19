import streamlit as st
from streamlit_option_menu import option_menu
from animation import *
from streamlit_lottie import st_lottie
import matplotlib.animation as animation
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle


#page congiguration
st.set_page_config(page_title= "Copper Modelling",
                   page_icon= 'random',
                   layout= "wide",)


#=========hide the streamlit main and footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

left,right=st.columns([1,3])

with left:
    url_link0="https://assets5.lottiefiles.com/packages/lf20_alg1vyxd.json"
    st_lottie = lottie_home1(url_link0)

with right:
    st.markdown("<h1 style='text-align: center; color: red;'>COPPER SELLING PRICE PREDICTION AND STATUS</h1>",
                unsafe_allow_html=True)

    selected = option_menu(None, ['HOME',"SELLING PRICE PREDICTION","STATUS","IMPROVISATION"],
                           icons=["house",'cash-coin','trophy'],orientation='horizontal',default_index=0)

    if selected=='HOME':
        with left:
            url_link1 = "https://assets1.lottiefiles.com/private_files/lf30_kit9njnq.json"
            st_lottie = lottie_home1(url_link1)

        st.write('## **WELCOME TO INDUSTRIAL COPPER MODELLING**')
        st.markdown('##### ***This project focuses on modelling industrial copper data using Python and various libraries such as pandas, numpy, scikit-learn. The objective of the project is to preprocess the data, handle missing values, detect outliers, and handle skewness. Additionally, regression and classification models will be built to predict the selling price and determine if a sale was won or lost. The trained models will be saved as a pickle file for later use in a Streamlit application.***')
        left ,right=st.columns([2,2])
        with left:
            st.write('### TECHNOLOGY USED')
            st.write('- PYTHON   (PANDAS, NUMPY)')
            st.write('- SCIKIT-LEARN')
            st.write('- DATA PREPROCESSING')
            st.write('- EXPLORATORY DATA ANALYSIS')
            st.write('- STREAMLIT')

        with right:
            st.write("### MACHINE LEARNING MODEL")
            st.write('#### REGRESSION - ***:red[EXTRA TREE REGRESSOR]***')
            st.write('- The ExtraTree Regressor is an ensemble learning method that belongs to the tree-based family of models.')
            st.write('#### CLASSIFICATION - ***:green[RANDOM FOREST CLASSIFIER]***')
            st.write('- The RandomForestClassifier is an ensemble learning method that combines multiple decision trees to create a robust and accurate classification model.')


    
    if selected == "SELLING PRICE PREDICTION":
        with left:
             url_link2 = "https://assets8.lottiefiles.com/packages/lf20_OPFirj1e4d.json"
             st_lottie = lottie_price1(url_link2)
        
        with right:
             
            st.markdown("# :blue[Predicting Results based on Trained Model]")
            # -----New Data inputs from the user for predicting the selling price-----
            a1 = st.text_input("Quantity (Min:611728 & Max:1722207579) ")
            b1 = st.text_input("Status (Enter 1 or 0)")
            c1 = st.text_input("Item Type (Enter 1 or 0)")
            d1 = st.text_input("Application (Min:2 & Max:1000)")
            e1 = st.text_input("Thickness (Min:1 & Max:300)")
            f1 = st.text_input("Width (Min:1, Max:2990)")
            g1 = st.text_input("Country (Min:10 & Max:100)")
            h1 = st.text_input("Customer (Min:12458 & Max:214748400)")
            i1 = st.text_input("Product Reference (Min:611728 & Max:1722207579)")
                    
            @st.cache_resource
            @st.cache_data

            import requests

            def download_file_from_google_drive(file_id, destination):
                URL = "https://drive.google.com/uc?id=" + file_id
                response = requests.get(URL)
                with open(destination, 'wb') as f:
                    f.write(response.content)
            
            # Replace 'your_file_id' with the actual file ID from the Google Drive link
            file_id = '12kwuGOOM8GXgxPnQsdk0fTGcx5JosKcH'
            download_file_from_google_drive(file_id, 'model.pkl')

            def load_model():
                with open('model.pkl', 'rb') as file:
                        model = pickle.load(file)
                return model

            regression_model = load_model()
            #st.write("Model loaded successfully.")

            # -----Submit Button for PREDICT RESALE PRICE-----   
            predict_button_1 = st.button("Predict Selling Price")

            if predict_button_1:
                try:

                    a1 = float(a1)
                    b1 = float(b1)
                    c1 = float(c1)
                    d1 = float(d1)
                    e1 = float(e1)
                    f1 = float(f1)
                    g1 = float(g1)
                    h1 = float(h1)
                    i1 = float(i1)

                    # -----Sending the user enter values for prediction to our model-----
                    new_sample_1 = np.array(
                            [[np.log(a1), b1, c1, d1, np.log(e1), f1, g1, h1, i1]])
                    new_pred_1 = regression_model.predict(new_sample_1)[0]
                    # Function to process the input
                        # Attempt to convert the input to float
                    
                    st.write(f'Predicted Selling Price : :green[₹] :green[{new_pred_1}]')
                except ValueError:
                    # Catch ValueError and display a user-friendly error message
                    st.error("Invalid input: Please enter a valid number.")

                st.info("The Predicted selling price may be differ from various reason like Supply and Demand Imbalances,Infrastructure and Transportation etc..",icon='ℹ️')



    if selected=='STATUS':
        with left:
             #url_link3 = "https://assets9.lottiefiles.com/packages/lf20_lw4olqnf.json"
             url_link3='https://assets8.lottiefiles.com/private_files/lf30_vsr6pvvl.json'
             lottie_status1(url_link3)
             url_link4= "https://assets1.lottiefiles.com/private_files/lf30_by9lgy8q.json"
             lottie_status1(url_link4)

        with right:
             st.write(
                    '##### ***<span style="color:yellow">Fill all the fields and Press the below button to view the status :red[WON / LOST] of copper in the desired time range</span>***',
                    unsafe_allow_html=True)

             cc1, cc2, cc3 = st.columns([2, 2, 2])
             with cc1:
                    quantity_cls = st.text_input('Enter Quantity  (Min:611728 & Max:1722207579) in tons')
                    thickness_cls = st.text_input('Enter Thickness (Min:0.18 & Max:400)')
                    width_cls= st.text_input('Enter Width  (Min:1, Max:2990)')

             with cc2:
                    selling_price_cls= st.text_input('Enter Selling Price  (Min:1, Max:100001015)')
                    item_cls = st.text_input('Item Type (Min:1 & Max: 6000)')
                    country_cls= st.text_input('Country Code (Min:10 & Max:100)')

             with cc3:
                    application_cls = st.text_input('Application Type (Min:2 & Max:1000)')
                    product_cls = st.text_input('Product Reference (Min:611728 & Max:1722207579)' )
                    customer_cls = st.text_input('Customer Number (Min:12458 & Max:214748400)')

                    
             with cc1:
                    st.write('')
                    st.write('')
                    st.write('')
                    if st.button('PREDICT STATUS'):
                        try:

                            @st.cache_resource
                            def load_model():
                                with open('classfier_model.pkl', 'rb') as file:
                                        model = pickle.load(file)
                                return model
                            
                            classification_model = load_model()

                            data_cls = [
                                    quantity_cls,
                                    thickness_cls,
                                    width_cls,
                                    selling_price_cls,
                                    application_cls,
                                    product_cls,
                                    customer_cls,
                                    item_cls,
                                    country_cls
                                        ]


                            x_cls = np.array(data_cls).reshape(1, -1)
                            new_pred_2 = classification_model.predict(x_cls)
                                    
                            if new_pred_2[0] ==1:
                                st.write(f'Predicted Status : :green[WON]')
                            else:
                                st.write(f'Predicted Status : :red[LOST]')
                        except ValueError:
                            st.error("Invalid input: Please enter a valid number.")

                    st.info("The Predicted Status may be differ from various reason like Supply and Demand Imbalances,Infrastructure and Transportation etc..",icon='ℹ️')


    if selected=='IMPROVISATION':
            import streamlit as st
            import pandas as pd
            import plotly.express as px
            import requests 
            from datetime import datetime
            from time import sleep        

            API_KEY = 'b97187410813c743eaf3fd4794364367'

            def fetch_live_copper_price(api_key):
                url = f"https://api.metalpriceapi.com/v1/latest?api_key={api_key}&base=USD&currencies=EUR,XAU,XAG"
                
                response = requests.get(url)
                data = response.json()
                copper_price = data['rates']['EUR']
                return copper_price 
            
            # Initialize data storage
            if 'price_data' not in st.session_state:
                st.session_state.price_data = []

            def update_data():
                copper_price = fetch_live_copper_price(API_KEY)
                timestamp = datetime.now()
                st.session_state.price_data.append({'timestamp': timestamp, 'copper_price': copper_price})
            
            # Layout of the Streamlit app
            st.title("Live Copper Dashboard")

            # Update data every minute
            st.button("Update Data", on_click=update_data)

            # Show latest copper price
            if st.session_state.price_data:
                latest_data = st.session_state.price_data[-1]
                st.metric(label="Current Copper Price (USD/ton)", value=f"${latest_data['copper_price']}")

            # Convert data to DataFrame for visualization
            if st.session_state.price_data:
                df = pd.DataFrame(st.session_state.price_data)

                # Plotting the copper price over time
                fig = px.scatter(df,size='copper_price', x='timestamp', y='copper_price', title='Live Copper Price Over Time')
                st.plotly_chart(fig)

            # Auto-refresh every 30 seconds
            if st.button("Start Auto-Refresh"):
                st.session_state.auto_refresh = True

            if st.button("Stop Auto-Refresh"):
                st.session_state.auto_refresh = False

            if 'auto_refresh' in st.session_state and st.session_state.auto_refresh:
                sleep(30)
                st.experimental_rerun()


            # Load sample data for improvisation insights
            improvisation_data = pd.read_csv('Copper_Final.csv')

            # Set up the Streamlit layout
            # Business Improvisation Section
            st.header("Business Improvisation Details")
            st.markdown("""
            ### Key Insights and Actions
            - **Copper Demand Analysis**: The demand for copper is projected to increase by 10% over the next year. Ensure adequate supply chain management to handle this increase.
            - **Cost Optimization**: Focus on optimizing production costs by improving efficiency in the manufacturing process.
            - **Market Trends**: Keep an eye on emerging markets, particularly in renewable energy sectors where copper usage is rising.
            """)       
            # Improvisation Data Insights
            st.subheader('Improvisation Data Insights')
            st.write(improvisation_data.head())

            # Additional Visualizations
            st.subheader('Market Trends Visualization')
            fig = px.line(improvisation_data, x='quantity tons_log', y='selling_price_log', title='Copper Market Price Trends')
            st.plotly_chart(fig)

            if st.checkbox('Show Correlation Heatmap'):
                st.subheader('Correlation Heatmap')
                corr = improvisation_data.corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
                st.plotly_chart(fig_corr)
