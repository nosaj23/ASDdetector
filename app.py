import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# import time
# import openai
# import joblib

# from skllm.config import SKLLMConfig
# from skllm.preprocessing import GPTSummarizer

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.pipeline import Pipeline

# import nltk
# from nltk.corpus import stopwords
# import spacy
# import contractions
# import string

# from wordcloud import WordCloud

#st.set_page_config(layout="wide") # Page expands to full width

#######################################################
# Initialize session state
#######################################################

# Q-CHAT Questions
Questions = [
    "Does your child look at you when you call his/her name?",
    "How easy is it for you to get eye contact with your child?",
    "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
    "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
    "Does your child pretend? (e.g care for dolls, talk on a toy phone)",
    "Does your child follow where you're looking?",
    "If you or someone in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g stroking hair, hugging them)",
    "Would you describe your child's first word?",
    "Does your child use simple gestures? (e.g wave goodbye)",
    "Does your child stare at nothing with no apparent purpose?"
    ]

# Q-CHAT Options
Q_Options_List = [["Always", "Usually", "Sometimes", "Rarely", "Never"],
    ["Very easy", "Quite easy", "Quite difficult", "Very difficult", "Impossible"],
    ["Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never"],
    ["Very typical", "Quite typical", "Slighly typical", "Very unusual", "My child doesn't speak"],        
    ]

#Q-CHAT Question & Option Mapping
Q_Options = [
    Q_Options_List[0], Q_Options_List[1], Q_Options_List[2], Q_Options_List[2], Q_Options_List[2], 
    Q_Options_List[2], Q_Options_List[0], Q_Options_List[3], Q_Options_List[2], Q_Options_List[2]
    ]

# Symptoms Dictionary
Symptoms_dict = {
    1 : "Abnormalities in eye contact",
    2 : "Deficits in nonverbal communication understanding",
    3 : "Difficulties in sharing imaginative play",
    4 : "Poorly integrated nonverbal communication",
    5 : "Deficits in social-emotional reciprocity",
    6 : "Poorly integrated verbal and nonverbal communication",
    7 : "Highly restricted, fixated interests that are abnormal in intensity or focus"
    }

Symptoms_map = [1,1,2,2,3,4,5,6,6,7]   #Q-CHAT Questions to Symptoms Mapping

# Symptoms Infographic
Symptoms_Info = {
    1  : ["A1-A2.png"],
    2  : ["A3-A4.png"],
    3  : ["A5.png"],
    4  : ["A6 1of2.png","A6 2of2.png"],
    5  : ["A7 1of2.png","A7 2of2.png"],
    6  : ["A8-A9 1of2.png",'A8-A9 2of2.png'],
    7  : ["A10.png"],
    99 : ['TD.png']
    }


# Header
st.title("Quantitative Checklist for Autism in Toddlers (Q-CHAT)")

st.warning("""
        **Important Disclaimer:**\n
        This tool is not intended to provide an official diagnosis and is merely a screening resource to assess the potential need for further professional diagnostic evaluation.
        """, icon="⚠️")

st.write("""
    The Q-CHAT (Quantitative Checklist for Autism in Toddlers) is designed for screening toddlers for signs of autism. 
    **The age range for the Q-CHAT is typically between 18 and 24 months (less than 4 years of age)**. It serves as a tool to identify 
    children who may benefit from a more detailed assessment for potential Autism Spectrum Disorders.
    """)

# First Initialization
if "page_no" not in st.session_state:
    st.session_state.page_no      = 0
    st.session_state.pred         = 0
    st.session_state.qchat_result = 0
    st.session_state.qchat_resp   = []
    st.session_state.bg_result    = []

#######################################################
# Function Name: load_pickle
# Description  : load pickle file
#######################################################
@st.cache_resource
def load_pickle():
    model = pickle.load(open('ada_hyper_f2.pkl', 'rb'))
    return model


#######################################################
# Function Name: get_yt_videos
# Description  : read YouTube Video Pool
#######################################################
@st.cache_data
def get_yt_videos():
    df = pd.read_csv('streamlit_yt_vids.csv', on_bad_lines='skip')
    video_dict = df.groupby('qid').apply(lambda group: dict(zip(group['id'], group['title']))).to_dict()
    return video_dict


#######################################################
# Function Name: get_yt_videos
# Description  : read YouTube Video Pool
#######################################################
def predict_data():
#        ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons',
#        'Sex', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test',
#        'Ethnicity_Latino', 'Ethnicity_Native Indian',
#        'Ethnicity_Others', 'Ethnicity_Pacifica', 'Ethnicity_White European',
#        'Ethnicity_asian', 'Ethnicity_black', 'Ethnicity_middle eastern',
#        'Ethnicity_mixed', 'Ethnicity_south asian']

    input_resp = st.session_state.qchat_resp.copy()
    model      = st.session_state.model

    bg_resp     = [0] * 15
    bg_resp[0]  = st.session_state.bg_result[0]                                          # Age_Mons
    bg_resp[1]  = 1 if st.session_state.bg_result[1] == "Male"            else 0         # Sex
    bg_resp[2]  = 1 if st.session_state.bg_result[3] == True              else 0         # Jaundice
    bg_resp[3]  = 1 if st.session_state.bg_result[4] == True              else 0         # Family_mem_with_ASD
    bg_resp[4]  = 1 if st.session_state.bg_result[5] == "Family Member"   else 0         # Who completed the test
    bg_resp[5]  = 1 if st.session_state.bg_result[2] == "Latino"          else 0         # Ethnicity_Latino
    bg_resp[6]  = 1 if st.session_state.bg_result[2] == "Native Indian"   else 0         # Ethnicity_Native Indian
    bg_resp[7]  = 1 if st.session_state.bg_result[2] == "Others"          else 0         # Ethnicity_Others
    bg_resp[8]  = 1 if st.session_state.bg_result[2] == "Pacifica"        else 0         # Ethnicity_Pacifica
    bg_resp[9]  = 1 if st.session_state.bg_result[2] == "White European"  else 0         # Ethnicity_White European
    bg_resp[10] = 1 if st.session_state.bg_result[2] == "Asian"           else 0         # Ethnicity_asian
    bg_resp[11] = 1 if st.session_state.bg_result[2] == "Black"           else 0         # Ethnicity_black
    bg_resp[12] = 1 if st.session_state.bg_result[2] == "Middle Eastern"  else 0         # Ethnicity_middle eastern
    bg_resp[13] = 1 if st.session_state.bg_result[2] == "Mixed"           else 0         # Ethnicity_mixed
    bg_resp[14] = 1 if st.session_state.bg_result[2] == "South Asian"     else 0         # Ethnicity_south asian

    input_resp.extend(bg_resp)
    input_data = np.array(input_resp).reshape(1, -1)
    prediction = model.predict(input_data)

    # 0 - Severe Symptoms
    # 1 - Mild Symptoms
    # 2 - Typically Developing

    return prediction

    
#######################################################
# Function Name: evaluate_Qchat
# Description  : Evaluate Qchat Scores
#######################################################
#@st.cache_data
def evaluate_Qchat(Responses, Q_Options):
    Answers = [0] * 10
    NoResponse = [i+1 for i, x in enumerate(Responses) if x is None]    # check for blank anwers

    for i, Response in enumerate(Responses):
        if i == 9: #Question 10
            Answers[i] = 1 if Response in Q_Options[i][0:3] else 0
        else:
            Answers[i] = 1 if Response in Q_Options[i][2:] else 0

    st.session_state.qchat_resp   = Answers
    st.session_state.qchat_result = sum(Answers)

    return NoResponse


#######################################################
# Function Name: demographics_questionnaire
# Description  : Show demographics questionnaire
#######################################################
def demographics_questionnaire():
    st.markdown("Please answer the following questions about your child")

    with st.form("Background Questionnaire"):
        st.info("##### Background Questions")

        B1 = st.number_input("B1. What age is your child (in months) [0-48]?", 0, 48, help="Select the age of your child in months.")
        B2 = st.radio("B2. What is your child's gender?", ["Male", "Female"], help="Choose the gender of your child.", horizontal=True, index=None)
        B3 = st.radio("B3. What is your child's ethnicity?", ["Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Native Indian", "Pacifica",
                                        "South Asian", "White European", "Mixed", "Others"],
                        help="Select the ethnicity that best describes your child.", horizontal=True, index=None)
        B4 = st.checkbox("B4. Has your child experienced jaundice?", help="Check this box if your child experienced jaundice")
        B5 = st.checkbox("B5. Do any of your child's immediate family members (siblings or parents) have a diagnosis of autism?", 
                          help="Check this box if any immediate family members have been diagnosed with autism")
        B6 = st.radio("B6. Who completed the test?", ["Family Member", "Health Care Professional"],
                        help="Select the person who administered the test.", horizontal=True, index=None)

        if B1 == 0: B1 = None
        Responses = [B1, B2, B3, B4, B5, B6]

        if st.form_submit_button("Submit"):
            NoResponse = [i+1 for i, x in enumerate(Responses) if x is None] 
            NoResponseText = ', '.join("B"+str(item) for item in NoResponse)  

            if len(NoResponse) > 0:
                st.error(f"Please answer questions {NoResponseText} then resubmit", icon="🚨")
                st.session_state.page_no = 0
            else:
                st.session_state.bg_result = Responses
                st.session_state.page_no += 1


#######################################################
# Function Name: qchat_questionnaire
# Description  : Show Q-CHAT-10 questionnaire
#######################################################
def qchat_questionnaire():
    st.markdown("Please answer the following questions about your child")

    with st.form("Q-CHAT Questionnaire"):
        st.info("##### Q-CHAT-10 Questions")

        Responses = [""] * 10
        for i, Question in enumerate(Questions):
            Responses[i] = st.radio(f"Q{i+1}. {Question}", Q_Options[i], index=None, horizontal=True) 

        if st.form_submit_button("Submit"):
            NoResponse = evaluate_Qchat(Responses, Q_Options)
            NoResponseText = ', '.join("Q"+str(item) for item in NoResponse)  

            if len(NoResponse) > 0:
                st.error(f"Please answer questions {NoResponseText} then resubmit", icon="🚨")
                st.session_state.page_no = 1
            else:
                st.session_state.page_no   += 1
    

#######################################################
# Function Name: show_results
# Description  : Show Results
#######################################################
def show_results():

    st.session_state.pred = predict_data()
    st.markdown("##### Q-CHAT Results")

    if st.session_state.pred == 0:
        st.info(f"""
                Score is **{st.session_state.qchat_result}** out of **10** \n
                The child has severe autism. Please refer your child for further professional diagnostic evaluation.
                """)
        
    elif st.session_state.pred == 1:
        st.info(f"""
                Score is **{st.session_state.qchat_result}** out of **10** \n
                The child has mild autism. Please refer your child for further professional diagnostic evaluation.
                """)
        
    else:
        st.info(f"""
                Score is **{st.session_state.qchat_result}** out of **10** \n
                The child is typically developing. No further action required.
                """)

    # if st.session_state.qchat_result > 3:
    #     st.info(f"""
    #             Score is **{st.session_state.qchat_result}** out of **10** \n
    #             Higher scores indicate stronger Autism Spectrum symptoms. Please refer your child for further professional diagnostic evaluation.
    #             """)
    # else:
    #     st.info(f"""
    #             Score is **{st.session_state.qchat_result}** out of **10** \n
    #             The child has no Autism Spectrum symptoms. No further action required.
    #             """)
        
        
#######################################################
# Function Name: get_yt_link
# Description  : format youtube link
#######################################################
def get_yt_link(video_id):
    return "https://www.youtube.com/watch?v="+video_id

#######################################################
# Function Name: show_yt_video
# Description  : Show Results
#######################################################
def show_yt_video(video_dict):

    st.info("Recommended Video to Watch")

    for video_id, title in video_dict.items():
        yt_column  = st.columns([0.5, 0.5])
        with yt_column[0]: st.video(get_yt_link(video_id))
        with yt_column[1]: st.write(title)

        # st.caption(title)
        # st.video(get_yt_link(video_id))


#######################################################
# MAIN
#######################################################
st.session_state.model = load_pickle()

if st.session_state.page_no == 0:
    demographics_questionnaire()

elif st.session_state.page_no == 1:
    qchat_questionnaire()

else:
    tab1, tab2 = st.tabs(["Results", "Recommendation"])

    with tab1:
        show_results()

    with tab2:
        Symptoms_Ids = set(Symptoms_map[i] for i, x in enumerate(st.session_state.qchat_resp) if x == 1)
        #Symptoms_Ids = (1,2,3,4)

        video_dict = get_yt_videos()

        # ------- Mild/Severe Autism -----#
        if st.session_state.pred < 2:
            expand_section = st.toggle("Expand All")
            st.info("Possible symptoms of the child")

            for i, Symptoms_Id in enumerate(Symptoms_Ids):
                with st.expander(f"Symptom {i+1}: {Symptoms_dict[Symptoms_Id]}", expanded=expand_section):

                    for image_name in Symptoms_Info[Symptoms_Id]:
                        st.image('images/' + image_name)

                    if Symptoms_Id in video_dict.keys():
                        show_yt_video(video_dict[Symptoms_Id])

        else:
            # ------- Typically Developing -----#
            for image_name in Symptoms_Info[99]:
                st.image('images/' + image_name)


                    # reco_column  = st.columns(2)
                    # with reco_column[0]: 
                    #     for image_name in Symptoms_Info[Symptoms_Id]:
                    #         st.image(image_name)
                    # with reco_column[1]:
                    #     if Symptoms_Id in video_dict.keys():
                    #         show_yt_video(video_dict[Symptoms_Id])
