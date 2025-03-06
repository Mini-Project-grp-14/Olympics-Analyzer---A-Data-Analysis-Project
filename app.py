import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import load
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# Load the dataset
df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Coding\python\learn streamlit\athlete_events.csv')
noc_df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Coding\python\learn streamlit\noc_regions.csv')
# Set page configuration
st.sidebar.title("")
st.sidebar.image(r"C:\Users\HP\OneDrive\Desktop\Coding\python\learn streamlit\5ef7e4d1131f087ddf4d8f41787a05ab-removebg-preview.png", width=200)
st.sidebar.title("Contents")
st.markdown("<br>", unsafe_allow_html=True)
landing_page_button = st.sidebar.button("Overall-Analysis")
country_analysis_button = st.sidebar.button("Country-wise Analysis")
sport_analysis_button = st.sidebar.button("Sport-wise Analysis")
Gold_Prediction_button = st.sidebar.button("Gold Prediction")
ideal_button = st.sidebar.button("Ideal Athlete")
simulation_button = st.sidebar.button("Simulation")

# Set a session state to track the current page
if 'page' not in st.session_state:
    st.session_state.page = 'Landing Page'  # Default to landing page

# Update page based on button clicks
if landing_page_button:
    st.session_state.page = 'Landing Page'
if country_analysis_button:
    st.session_state.page = 'Country-wise Analysis'
if sport_analysis_button:
    st.session_state.page = 'Sport-wise Analysis'
if Gold_Prediction_button:
    st.session_state.page = 'Gold Prediction'
if simulation_button:
    st.session_state.page = 'Simulation'
if ideal_button:
    st.session_state.page = 'Ideal_Athlete'

# Landing Page
if st.session_state.page == "Landing Page":
    # Centralize header and add spacing
    st.markdown("""
        <div style="text-align:center; margin-bottom: 50px;">
            <h1>🏅 Olympic Data Analyser-Overall Analysis</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Use columns for placing metrics and align them centrally
    st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 50px;">
            <div style="margin-right: 50px;">
                <h2><b>No. of Olympic Sessions</b></h2>
                <center><h3>{}</h3><center>
            </div>
            <div style="margin-right: 50px;">
                <h2><b>No. of Games Played</b></h2>
                <center><h3>{}</h3><center>
            </div>
            <div>
                <h2><b>Total Number of Participants</b></h2>
                <center><h3>{}</h3><center>
            </div>
        </div>
    """.format(df['Year'].nunique(), df['Event'].nunique(), df['ID'].nunique()), unsafe_allow_html=True)

    # Add spacing before the graph
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Center align the Men vs Women graph
    st.markdown("<h1 style='text-align: center;'>Men Vs Women Participation Analysis</h1>", unsafe_allow_html=True)
    gender_counts = df['Sex'].value_counts().reset_index()
    gender_counts.columns = ['Sex', 'Count']

    # Create the plot
    fig_gender = px.bar(gender_counts, x='Sex', y='Count', color='Sex',
                        labels={"Count": "Number of Participants"},
                        color_discrete_map={'Male': 'blue', 'Female': 'pink'})

    # Adjust plot size and align it centrally
    fig_gender.update_layout(height=500, width=700)
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gender)
    st.markdown("</div>", unsafe_allow_html=True)

    # Add more spacing between sections
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Medal Tally Count Section
    st.markdown("<h1 style='text-align: center;'>Medal Tally Count</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>(Country with Maximum Medals)</h1>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    max_gold = df[df['Medal'] == 'Gold']['NOC'].value_counts().idxmax()
    max_silver = df[df['Medal'] == 'Silver']['NOC'].value_counts().idxmax()
    max_bronze = df[df['Medal'] == 'Bronze']['NOC'].value_counts().idxmax()

    st.markdown(f"""
        <style>
            .podium-container {{
                display: flex;
                justify-content: center;
                align-items: flex-end;
                margin: 50px 0;
                height: 450px; /* Overall height */
            }}
            .podium {{
                background-color: #f5f5f5;
                border-radius: 0; /* Remove rounded corners */
                box-shadow: none; /* Remove shadow */
                text-align: center;
                position: relative;
                margin: 2px; /* Margin between podiums */
            }}
            .gold {{
                border: 2px solid black;
                background-color: gold;
                height: 360px; /* Height for gold */
                width: 240px; /* Width for gold */
                z-index: 3; /* Ensure gold is on top */
            }}
            .silver {{
                border: 2px solid black;
                background-color: silver;
                height: 270px; /* Height for silver */
                width: 180px; /* Width for silver */
                z-index: 1; /* Ensure silver is in the middle */
            }}
            .bronze {{
                border: 2px solid black;
                background-color: #cd7f32;
                height: 180px; /* Height for bronze */
                width: 150px; /* Width for bronze */
                z-index: 1; /* Ensure bronze is at the bottom */
            }}
            .podium-title {{
                position: absolute;
                top: 5px; /* Position at the top of the podium */
                left: 50%;
                transform: translateX(-50%);
                font-weight: bold;
                z-index: 4; /* Ensure text is on top */
                font-size: 20px; /* Font size for country name */
            }}
            .podium-text {{
                position: absolute;
                bottom: 20px; /* Position the count near the bottom of the podium */
                left: 50%;
                transform: translate(-50%, 0);
                font-weight: bold;
                z-index: 4; /* Ensure text is on top */
                font-size: 24px; /* Font size for medal count */
            }}
        </style>

        <div class="podium-container">
            <div class="podium silver" style="margin-right: 3px;">
                <div class="podium-title">{max_silver}</div> <!-- Country name on podium -->
                <div class="podium-text">Silver Medals</div>
            </div>
            <div class="podium gold" style="margin-left: 3px; margin-right: 3px;">
                <div class="podium-title">{max_gold}</div> <!-- Country name on podium -->
                <div class="podium-text">Gold Medals</div>
            </div>
            <div class="podium bronze" style="margin-left: 3px;">
                <div class="podium-title">{max_bronze}</div> <!-- Country name on podium -->
                <div class="podium-text">Bronze Medals</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Add spacing before athlete section
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Table
    st.markdown("<h1 style='text-align: center;'>Top 10 Athletes with Maximum Gold Medals</h1>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    gold_medals = df[df['Medal'] == 'Gold']
    top_athletes = gold_medals.groupby(['Name', 'Sport']).size().nlargest(10).reset_index(name='Gold Medals')

    # Center table
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.table(top_athletes)
    st.markdown("</div>", unsafe_allow_html=True)

# Country-wise Analysis Page
elif st.session_state.page == "Country-wise Analysis":
    st.markdown("""
            <div style="text-align:center; margin-bottom: 50px;">
                <h1>🏅 Olympic Data Analyser-Country-wise Analysis</h1>
            </div>
        """, unsafe_allow_html=True)
    df = pd.merge(df, noc_df[['NOC', 'region']], on='NOC', how='left')
    # Get unique countries for the dropdown
    countries = df['region'].unique()
    selected_country = st.selectbox("Select a Country", countries)

    # Filter the dataset for the selected country
    country_data = df[df['region'] == selected_country]

    # Calculate total medals
    total_gold = country_data[country_data['Medal'] == 'Gold'].shape[0]
    total_silver = country_data[country_data['Medal'] == 'Silver'].shape[0]
    total_bronze = country_data[country_data['Medal'] == 'Bronze'].shape[0]
    total_medals = total_gold + total_silver + total_bronze
    total_participants = country_data['ID'].nunique()

    # Display the metrics
    st.markdown(
        f"**Total Medals:** {total_medals} (Gold: {total_gold}, Silver: {total_silver}, Bronze: {total_bronze})")
    st.markdown(f"**Total Participants:** {total_participants}")

    # Medal count by sport
    medals_by_sport = country_data[country_data['Medal'].notnull()]['Sport'].value_counts().reset_index()
    medals_by_sport.columns = ['Sport', 'Medal Count']

    # Plot the medal count by sport
    fig = px.bar(medals_by_sport, x='Medal Count', y='Sport', title=f"Medals Won by {selected_country} by Sport",
                 color='Medal Count')
    st.plotly_chart(fig)

    # Participation trend over the years
    participation_trend = country_data['Year'].value_counts().sort_index().reset_index()
    participation_trend.columns = ['Year', 'Participants']

    # Plot the participation trend
    fig_trend = px.line(participation_trend, x='Year', y='Participants',
                        title=f"Participation Trend of {selected_country} Over the Years")
    st.plotly_chart(fig_trend)

    # Top 5 athletes from the selected country
    top_athletes = country_data[country_data['Medal'] == 'Gold'] \
        .groupby(['Name', 'Sport']) \
        .size() \
        .nlargest(5) \
        .reset_index(name='Gold Medals')

    # Display the top 5 athletes
    st.markdown("<h2 style='text-align: center;'>Top 5 Athletes from {}</h2>".format(selected_country),
                unsafe_allow_html=True)
    st.table(top_athletes)

elif st.session_state.page == "Sport-wise Analysis":
    # Load NOC to country mapping

    # Merge the datasets to get full country names
    df = pd.merge(df, noc_df[['NOC', 'region']], on='NOC', how='left')
    # Title and sport selection
    st.markdown("<h1 style='text-align: center;'>🏅 Olympic Data Analyser - Sport Wise Analysis</h1>",
                unsafe_allow_html=True)

    # Sport selector
    sport = st.selectbox('Select a Sport:', df['Sport'].unique())

    # Separate data for the selected sport
    sport_data = df[df['Sport'] == sport]

    # Separate data for men and women
    men_data = sport_data[sport_data['Sex'] == 'M']
    women_data = sport_data[sport_data['Sex'] == 'F']

    # Number of participants (Men and Women)
    st.markdown(f"<h2 style='text-align: center;'>Participants in {sport}</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    col1.metric(label="Men Participants", value=men_data['ID'].nunique())
    col2.metric(label="Women Participants", value=women_data['ID'].nunique())

    # Medal Distribution in Table Format (Men and Women)
    st.markdown(f"<h2 style='text-align: center;'>Medal Distribution in {sport}</h2>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    # Medal distribution for men
    try:
        men_medal_distribution = men_data.groupby(['region', 'Medal']).size().unstack(fill_value=0).reset_index()
        col3.markdown("<h3 style='text-align: center;'>Men</h3>", unsafe_allow_html=True)
        col3.table(men_medal_distribution[['region', 'Gold', 'Silver', 'Bronze']].rename(columns={'region': 'Country'}))
    except Exception as e:
        col3.error("Not enough data available for men.")

    # Medal distribution for women
    try:
        women_medal_distribution = women_data.groupby(['region', 'Medal']).size().unstack(fill_value=0).reset_index()
        col4.markdown("<h3 style='text-align: center;'>Women</h3>", unsafe_allow_html=True)
        col4.table(
            women_medal_distribution[['region', 'Gold', 'Silver', 'Bronze']].rename(columns={'region': 'Country'}))
    except Exception as e:
        col4.error("Not enough data available for women.")

    # Top 5 athletes (Men and Women)
    st.markdown(f"<h2 style='text-align: center;'>Top 5 Athletes in {sport}</h2>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)

    # Top 5 men athletes
    try:
        top_men_athletes = men_data[men_data['Medal'] == 'Gold'].groupby('Name').size().nlargest(5).reset_index(
            name='Gold Medals')
        col5.markdown("<h3 style='text-align: center;'>Men</h3>", unsafe_allow_html=True)
        col5.table(top_men_athletes)
    except Exception as e:
        col5.error("Not enough data available for top men athletes.")

    # Top 5 women athletes
    try:
        top_women_athletes = women_data[women_data['Medal'] == 'Gold'].groupby('Name').size().nlargest(5).reset_index(
            name='Gold Medals')
        col6.markdown("<h3 style='text-align: center;'>Women</h3>", unsafe_allow_html=True)
        col6.table(top_women_athletes)
    except Exception as e:
        col6.error("Not enough data available for top women athletes.")

    # Top 3 performing nations (Men and Women)
    st.markdown(f"<h2 style='text-align: center;'>Top 3 Performing Nations in {sport}</h2>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)

    # Top 3 men nations with country names
    try:
        top_men_nations = men_data.groupby('region').size().nlargest(3).reset_index(name='Medals')
        col7.markdown("<h3 style='text-align: center;'>Men</h3>", unsafe_allow_html=True)
        col7.table(top_men_nations[['region', 'Medals']].rename(columns={'region': 'Country'}))
    except Exception as e:
        col7.error("Not enough data available for top men nations.")

    # Top 3 women nations with country names
    try:
        top_women_nations = women_data.groupby('region').size().nlargest(3).reset_index(name='Medals')
        col8.markdown("<h3 style='text-align: center;'>Women</h3>", unsafe_allow_html=True)
        col8.table(top_women_nations[['region', 'Medals']].rename(columns={'region': 'Country'}))
    except Exception as e:
        col8.error("Not enough data available for top women nations.")

elif st.session_state.page == "Gold Prediction":
    st.markdown("<h1 style='text-align: center;'>🏅 Olympic Data Analyser - Gold Prediction</h1>",
                unsafe_allow_html=True)
    rf_classifier = load('rf_classifier.joblib')

    # Load your dataset
    model_data = pd.read_csv(
        r'C:\Users\HP\OneDrive\Desktop\Coding\python\learn streamlit\athlete_events.csv')  # Replace with your actual dataset path

    # Preprocess the data for encoding
    label_encoders = {}
    categorical_cols = model_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        model_data[col] = le.fit_transform(model_data[col])
        label_encoders[col] = le  # Save the encoder

    # User input for Country (Team)
    teams = label_encoders['Team'].inverse_transform(np.arange(len(label_encoders['Team'].classes_)))
    selected_team = st.selectbox("Select a Country/Team", teams)

    # Ensure the team is properly encoded
    if selected_team in label_encoders['Team'].classes_:
        encoded_team = label_encoders['Team'].transform([selected_team])[0]
    else:
        st.error(f"Team '{selected_team}' not found in training data. Unable to provide predictions.")
        st.stop()  # Stop the execution if team is not found

    # Filter data for the selected team
    team_data = model_data[model_data['Team'] == encoded_team]

    # Check if the 'Sport' column exists
    if 'Sport' in team_data.columns:

        # Get unique sports the selected team participated in
        sports = team_data['Sport'].unique()

        # Prepare predictions for each sport
        sport_predictions = []
        for sport in sports:
            # Filter data for the sport
            sport_data = team_data[team_data['Sport'] == sport]

            # Get the average of numeric features like 'Age', 'Height', 'Weight'
            mean_age = sport_data['Age'].mean() if not sport_data.empty else np.nan
            mean_height = sport_data['Height'].mean() if not sport_data.empty else np.nan
            mean_weight = sport_data['Weight'].mean() if not sport_data.empty else np.nan

            # Prepare a row for prediction (with encoded values)
            sport_row = pd.DataFrame({
                'Team': [label_encoders['Team'].transform([selected_team])[0]],
                'Sport': [sport],  # This remains as the encoded value
                'Age': [mean_age if not np.isnan(mean_age) else 0],
                'Height': [mean_height if not np.isnan(mean_height) else 0],
                'Weight': [mean_weight if not np.isnan(mean_weight) else 0]
            })

            # Ensure the feature order matches the model's training data
            columns_to_drop = ['Gold_Won', 'Medal']
            existing_columns = [col for col in columns_to_drop if col in model_data.columns]

            sport_row = sport_row.reindex(columns=model_data.drop(columns=existing_columns).columns,
                                          fill_value=0)

            # Predict the probability of winning a gold medal
            gold_prob = rf_classifier.predict_proba(sport_row)[:, 1][0] if not sport_row.empty else 0

            # Append to the predictions list
            sport_predictions.append({
                'Sport': label_encoders['Sport'].inverse_transform([sport_row['Sport'][0]])[0],
                # Inverse transform to get the sport name
                'Gold Probability': gold_prob*100
            })

        # Convert the list of predictions to a DataFrame for display
        predictions_df = pd.DataFrame(sport_predictions)

        # Sort the DataFrame by 'Gold Probability' in descending order
        predictions_df = predictions_df.sort_values(by='Gold Probability', ascending=False)

        # Drop the index and reset the DataFrame
        predictions_df.reset_index(drop=True, inplace=True)

        # Rename columns for clarity (optional)
        predictions_df.columns = ['Sport', 'Gold Probability']

        # Display the results in a table format
        st.markdown(f"<h3>Gold Medal Probability for {selected_team}</h3>", unsafe_allow_html=True)
        st.table(predictions_df)


elif st.session_state.page == "Ideal_Athlete":
    st.markdown("<h1 style='text-align: center;'>🏅 Olympic Data Analyser-  Ideal Athlete Profile</h1>",
                unsafe_allow_html=True)
    selected_sport = st.selectbox("Select a Sport", sorted(df['Sport'].unique()))

    # Filter dataset based on selected sport
    sport_data = df[df['Sport'] == selected_sport]

    # Split by gender
    male_data = sport_data[sport_data['Sex'] == 'M']
    female_data = sport_data[sport_data['Sex'] == 'F']


    # Function to preprocess and normalize the data
    def preprocess_data(data):
        # Drop rows with missing values
        data = data.dropna(subset=['Height', 'Weight', 'Age'])

        if data.empty:
            return None, None

        # Extract only relevant columns
        features = data[['Height', 'Weight', 'Age']]

        # Normalize the features using StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        return features_scaled, scaler


    # Function to apply K-Means clustering and get the ideal athlete profile
    def cluster_athletes(data, gender):
        features_scaled, scaler = preprocess_data(data)

        if features_scaled is None:
            st.write(f"Not enough data available for {gender} athletes in {selected_sport}.")
            return None

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=0)  # We use 3 clusters here as an example
        kmeans.fit(features_scaled)

        # Get the centroid of the largest cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        largest_cluster_idx = labels[np.argmax(counts)]
        ideal_profile_scaled = kmeans.cluster_centers_[largest_cluster_idx]

        # Convert the scaled ideal profile back to the original scale
        ideal_profile = scaler.inverse_transform([ideal_profile_scaled])[0]

        return ideal_profile


    # Function to find closest athlete to the ideal profile
    def find_closest_athlete(data, ideal_profile):
        data_filtered = data.dropna(subset=['Height', 'Weight', 'Age'])
        if data_filtered.empty:
            return None
        ideal_df = pd.DataFrame([ideal_profile], columns=['Height', 'Weight', 'Age'])
        distances = cdist(data_filtered[['Height', 'Weight', 'Age']], ideal_df)
        data_filtered['Distance'] = distances
        closest_athlete = data_filtered.loc[data_filtered['Distance'].idxmin()]
        return closest_athlete


    # Display ideal attributes and closest athlete for males
    st.subheader(f"Ideal Male Athlete for {selected_sport}")
    ideal_male = cluster_athletes(male_data, 'male')

    if ideal_male is not None:
        st.table(pd.DataFrame({
            'Attribute': ['Height (cm)', 'Weight (kg)', 'Age (years)'],
            'Ideal Male Athlete Profile': [f"{ideal_male[0]:.2f}", f"{ideal_male[1]:.2f}", f"{ideal_male[2]:.2f}"]
        }))

        # Find closest male athlete
        closest_male = find_closest_athlete(male_data, ideal_male)
        if closest_male is not None:
            st.subheader("Closest Male Athlete")
            st.table(pd.DataFrame({
                'Attribute': ['Name', 'Team', 'Height (cm)', 'Weight (kg)', 'Age (years)'],
                'Athlete': [closest_male['Name'], closest_male['Team'], closest_male['Height'],
                            closest_male['Weight'], closest_male['Age']]
            }))
        else:
            st.write("Not enough data to find the closest male athlete.")

    # Display ideal attributes and closest athlete for females
    st.subheader(f"Ideal Female Athlete for {selected_sport}")
    ideal_female = cluster_athletes(female_data, 'female')

    if ideal_female is not None:
        st.table(pd.DataFrame({
            'Attribute': ['Height (cm)', 'Weight (kg)', 'Age (years)'],
            'Ideal Female Athlete Profile': [f"{ideal_female[0]:.2f}", f"{ideal_female[1]:.2f}",
                                             f"{ideal_female[2]:.2f}"]
        }))

        # Find closest female athlete
        closest_female = find_closest_athlete(female_data, ideal_female)
        if closest_female is not None:
            st.subheader("Closest Female Athlete")
            st.table(pd.DataFrame({
                'Attribute': ['Name', 'Team', 'Height (cm)', 'Weight (kg)', 'Age (years)'],
                'Athlete': [closest_female['Name'], closest_female['Team'], closest_female['Height'],
                            closest_female['Weight'], closest_female['Age']]
            }))
        else:
            st.write("Not enough data to find the closest female athlete.")

    # Extra Feature: Median age of medal winners
    medal_winners = sport_data[sport_data['Medal'].notna()]
    if not medal_winners.empty:
        median_age_medalists = medal_winners['Age'].median()
        st.subheader("Median Age of Medalists")
        st.write(f"The median age of medalists in {selected_sport} is {median_age_medalists:.2f} years.")
    else:
        st.write(f"No medalist data available for {selected_sport}.")

elif st.session_state.page == "Simulation":
    st.markdown("<h1 style='text-align: center;'>🏅 Olympic Data Analyser - Medal Race Simulation</h1>",
                unsafe_allow_html=True)
    model_data = pd.read_csv(
        r'C:\Users\HP\OneDrive\Desktop\Coding\python\learn streamlit\athlete_events.csv')  # Update with your actual data file path
    label_encoders = {
        'Team': LabelEncoder(),
        'Sport': LabelEncoder(),
    }

    # Fit label encoders on the data
    label_encoders['Team'].fit(model_data['Team'])
    label_encoders['Sport'].fit(model_data['Sport'])

    # Get unique country names and sports
    countries = label_encoders['Team'].inverse_transform(np.arange(len(label_encoders['Team'].classes_)))
    sports = label_encoders['Sport'].inverse_transform(np.arange(len(label_encoders['Sport'].classes_)))

    # Select countries
    selected_countries = st.multiselect("Select Countries", countries)

    # Number of simulations
    num_simulations = st.number_input("Number of Simulations", min_value=1, value=1000)


    # Button to run the simulation

    def calculate_medal_probabilities(model_data):
        medal_probabilities = {}

        for country in selected_countries:
            # Filter the data for the selected country
            country_data = model_data[model_data['Team'] == country]

            # Calculate medal probabilities for each sport
            sport_probabilities = {}
            for sport in sports:
                sport_data = country_data[country_data['Sport'] == sport]
                total_events = len(sport_data)
                if total_events > 0:
                    gold_wins = len(sport_data[sport_data['Medal'] == 'Gold'])
                    silver_wins = len(sport_data[sport_data['Medal'] == 'Silver'])
                    bronze_wins = len(sport_data[sport_data['Medal'] == 'Bronze'])

                    # Calculate probabilities
                    sport_probabilities[
                        sport] = gold_wins / total_events  # Example: Using gold wins to calculate probability

            medal_probabilities[country] = sport_probabilities

        return medal_probabilities


    def simulate_medals(probabilities, num_simulations=1000):
        results = {}

        for country, sports in probabilities.items():
            results[country] = {'Gold': 0, 'Silver': 0, 'Bronze': 0}

            for _ in range(num_simulations):
                gold_count = 0
                silver_count = 0
                bronze_count = 0

                for sport, prob in sports.items():
                    if prob > 0:  # Only simulate if there's a positive probability
                        rand_val = np.random.rand()
                        if rand_val < prob:
                            gold_count += 1
                        elif rand_val < prob + (0.5 * prob):
                            silver_count += 1
                        elif rand_val < prob + (0.75 * prob):
                            bronze_count += 1

                results[country]['Gold'] += gold_count
                results[country]['Silver'] += silver_count
                results[country]['Bronze'] += bronze_count

        # Calculate averages
        for country in results:
            results[country]['Gold'] /= num_simulations
            results[country]['Silver'] /= num_simulations
            results[country]['Bronze'] /= num_simulations

        return results


    if st.button("Run Simulation"):
        if selected_countries:
            # Calculate medal probabilities
            medal_probabilities = calculate_medal_probabilities(model_data)

            # Run simulations
            medal_results = simulate_medals(medal_probabilities, num_simulations)

            # Display results
            results_df = pd.DataFrame(medal_results).T
            results_df.columns = ['Gold', 'Silver', 'Bronze']

            results_df = results_df.round(0).astype(int)
            st.markdown(
                """
                <style>
                .dataframe {
                    width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.table(results_df)

        else:
            st.write("Please select at least one country.")


