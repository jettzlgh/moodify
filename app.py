import streamlit as st
from spotipy.oauth2 import SpotifyOAuth
import spotipy

# Spotify API credentials
CLIENT_ID = 'a8d4240370924001b35b6449eab1c389'
CLIENT_SECRET = '8e4d28a567454e6db73d3a856121feaa'
REDIRECT_URI = 'http://localhost:8501/'
# Define the scope of access
SCOPE = 'user-library-read playlist-read-private'

# Initialize Spotify OAuth
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE)

# Apply Spotify Branding
st.markdown("""
    <style>
    /* General Layout */
    .main {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Sidebar */
    .stSidebar {
        background-color: #121212;
        color: #FFFFFF;
    }

    /* Spotify Login Button */
    .spotify-btn {
        background-color: #1ED760;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 18px;
        cursor: pointer;
        text-decoration: none;
    }

    .spotify-btn:hover {
        background-color: #1DB954;
    }

    /* Centering Login Section */
    .login-section {
        text-align: center;
        margin-top: 50px;
    }

    /* Playlist Header */
    .playlist-header {
        color: #1ED760;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Playlist Items */
    .playlist-item {
        background-color: #121212;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }

    .playlist-item img {
        width: 60px;
        height: 60px;
        border-radius: 8px;
        margin-right: 15px;
    }

    .playlist-item h4 {
        margin: 0;
        color: #FFFFFF;
        font-size: 18px;
    }

    .profil-picture {
        width: 100%;
        height: auto;
        border-radius: 10px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }


     /* Days Section */
    .day-item {
        background-color: #282828;
        color: #FFFFFF;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        width: 200px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin: 10px;
    }
    .day-item img {
        width: 100%;
        height: auto;
        border-radius: 10px;
    }
    .day-item-title {
        color: #1ED760;
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
    }
    .day-item-description {
        color: #1ED760;
        font-size: 16px;
        margin-top: 10px;
    }
    .days-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# Spotify Login Section
st.markdown("<h1 style='color:#1ED760;text-align:center;font-size:65px;'>Welcome to MOODIFY</h1>", unsafe_allow_html=True)


auth_code = st.experimental_get_query_params().get('code')

# If the user is not logged in, show the login button
if not auth_code:
    auth_url = sp_oauth.get_authorize_url()
    st.markdown("<p style='color:#1ED760;font-size:24px;text-align:center;'>Log in to explore your playlists and music data.</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='login-section'><a href='{auth_url}' class='spotify-btn'>Login to Spotify</a></div>", unsafe_allow_html=True)
else:
    # Handle Redirect and Authentication
    token_info = sp_oauth.get_access_token(auth_code[0])
    sp = spotipy.Spotify(auth=token_info['access_token'])

    # User Info Section
    user_info = sp.current_user()
    st.markdown(f"<h2 style='color:#1DB954;text-align:center;'>Welcome, {user_info['display_name']}!</h2>", unsafe_allow_html=True)
    if user_info.get('images') and len(user_info['images']) > 0:
        st.image(user_info['images'][0]['url'], width=100)
        # st.markdown(f"""
        # <div class='profil-picture'>
        #     <img src='{user_info['images'][0]['url']}'>
        # </div>
        # """, unsafe_allow_html=True)

    # Playlist Section
    st.markdown("<h3 style:'text-align:center;' class='playlist-header'>Your Playlists</h3>", unsafe_allow_html=True)
    playlists = sp.current_user_playlists()
    print(playlists['items'][2])
    print('yoyo')
    #for playlist in playlists['items']:
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"""
        <div class='playlist-item'>
            <img src='{playlists['items'][0]['images'][0]['url']}' alt='Playlist Cover'>
            <h4>{playlists['items'][0]['name']} ({playlists['items'][0]['tracks']['total']} tracks)</h4>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='playlist-item'>
            <img src='{playlists['items'][1]['images'][0]['url']}' alt='Playlist Cover'>
            <h4>{playlists['items'][1]['name']} ({playlists['items'][1]['tracks']['total']} tracks)</h4>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='playlist-item'>
            <img src='{playlists['items'][2]['images'][0]['url']}' alt='Playlist Cover'>
            <h4>{playlists['items'][2]['name'][:15]} ({playlists['items'][2]['tracks']['total']} tracks)</h4>
        </div>
        """, unsafe_allow_html=True)


# Define the items to display
    days_of_week = [
        {"day": "Monday", "image": "https://static.vecteezy.com/system/resources/previews/048/975/865/non_2x/fashion-male-model-in-matte-gray-headphones-with-neon-purple-background-stylish-lifestyle-portrait-for-contemporary-males-fashion-free-photo.jpg", "text": "Feeling Down"},
        {"day": "Tuesday", "image": "https://static.vecteezy.com/system/resources/previews/048/976/373/non_2x/fashion-male-model-in-neon-green-headphones-with-neon-red-lifestyle-dynamic-and-bold-males-portrait-free-photo.jpg", "text": "Energy burning"},
        {"day": "Wednesday", "image": "https://static.vecteezy.com/system/resources/previews/048/976/653/non_2x/stylish-male-fashion-male-model-in-neon-green-headphones-with-neon-red-background-trendy-urban-portrait-featuring-modern-appeal-free-photo.jpg", "text": "Energy burning"},
        {"day": "Thursday", "image": "https://static.vecteezy.com/system/resources/thumbnails/048/974/544/small/handsome-fashion-male-model-in-males-headphones-with-neon-green-showcased-in-a-modern-lifestyle-portrait-background-free-photo.jpg", "text": "Deep Working"},
        {"day": "Friday", "image": "https://static.vecteezy.com/system/resources/thumbnails/048/976/564/small/stylish-male-model-in-neon-green-headphones-with-neon-yellow-background-trendy-urban-fashion-statemalet-with-a-modern-twist-free-photo.jpg", "text": "Feel-Good"},
        {"day": "Saturday", "image": "https://static.vecteezy.com/system/resources/thumbnails/048/974/191/small/elegant-male-model-in-black-headphones-with-neon-aqua-highlights-showcased-in-a-contemporary-lifestyle-portrait-setting-free-photo.jpg", "text": "Moving Fast"},
        {"day": "Sunday", "image": "https://via.placeholder.com/150", "text": "Slow-Down Sunday"}
    ]


    st.markdown("<div class='days-row'>", unsafe_allow_html=True)
    # Display the items in rows
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"""
        <div class='day-item'>
            <img src='{days_of_week[0]['image']}' alt='{days_of_week[0]['day']}'>
            <p class='day-item-title'>{days_of_week[0]['day']}</p>
            <p class='day-item-description'>{days_of_week[0]['text']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='day-item'>
            <img src='{days_of_week[1]['image']}' alt='{days_of_week[1]['day']}'>
            <p class='day-item-title'>{days_of_week[1]['day']}</p>
            <p class='day-item-description'>{days_of_week[1]['text']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='day-item'>
            <img src='{days_of_week[2]['image']}' alt='{days_of_week[2]['day']}'>
            <p class='day-item-title'>{days_of_week[2]['day']}</p>
            <p class='day-item-description'>{days_of_week[2]['text']}</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"""
        <div class='day-item'>
            <img src='{days_of_week[3]['image']}' alt='{days_of_week[3]['day']}'>
            <p class='day-item-title'>{days_of_week[3]['day']}</p>
            <p class='day-item-description'>{days_of_week[3]['text']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='day-item'>
            <img src='{days_of_week[4]['image']}' alt='{days_of_week[4]['day']}'>
            <p class='day-item-title'>{days_of_week[4]['day']}</p>
            <p class='day-item-description'>{days_of_week[4]['text']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='day-item'>
            <img src='{days_of_week[5]['image']}' alt='{days_of_week[5]['day']}'>
            <p class='day-item-title'>{days_of_week[5]['day']}</p>
            <p class='day-item-description'>{days_of_week[5]['text']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
