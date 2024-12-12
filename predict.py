import streamlit as st
import pandas as pd
import joblib
from sklearn.exceptions import NotFittedError

# Memproses the encoder dictionary, scaler, dan model
encoder_dict = joblib.load('encoder_dict.pkl')  # Gunakan dictionary encoder
scaler = joblib.load('scaler.pkl')  # Pastikan scaler.pkl sudah ada
model = joblib.load('predict_salary_model.pkl')  # Pastikan predict_salary_model.pkl sudah ada

st.markdown("""
<style>
    .stApp {
  background-color: #DEECFF; /* Warna latar belakang utama */
}

.stTitle {
  color: #3498db; /* Biru judul */
}

.stButton {
  background-color: #4CAF50; /* Hijau tombol */
  color: white;
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

.stError {
  background-color: #f8d7da; /* Merah muda pesan error */
  color: #721c24;
  padding: 10px;
  border: 1px solid #f5c6cb;
  border-radius: 4px;
}
</style>
""", unsafe_allow_html=True) # menampilkan konten HTML

# Fungsi untuk melakukan preprocessing input pengguna
def preprocess_input(age, gender, education_level, job_title, years_experience):
    try:
        # Buat DataFrame dengan input pengguna
        data = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Education Level': [education_level],
                             'Job Title': [job_title], 'Years of Experience': [years_experience]})

        # Encoding menggunakan encoder yang telah disimpan
        categorical_cols = ['Gender', 'Education Level', 'Job Title'] #Menyimpan daftar kolom yang berisi data kategorikal  
        for col in categorical_cols: #perulangan setiap kolom kategorikal.
            if col in encoder_dict: # Memeriksa kolom tersebut memiliki encoder yang dilatih
                # Tangani jika nilai input tidak ditemukan di encoder
                try:
                    data[col] = encoder_dict[col].transform(data[col]) # mengubah nilai kolom -> representasi numerik
                except ValueError as e:
                    st.error(f"Error encoding {col}: {e}")
                    return None
        # Scaling (gunakan scaler yang sudah dilatih sebelumnya)
        scaled_data = scaler.transform(data) #melakukan penskalaan pada seluruh data.

        return scaled_data

    except NotFittedError as e: # akan muncul jika scaler belum dilatih
        st.error(f"Scaler atau encoder belum dilatih: {e}")
        return None

# Fungsi utama aplikasi
def main():
    # Title and description
    st.title("Salary Prediction App")
    st.write("This app predicts salary based on Age, Gender, Education Level, Job Title, and Years of Experience.")

    # Input fields di body (bukan sidebar)
    age = st.number_input("Age", min_value=18, max_value=65, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    job_title = st.selectbox("Job Title", ['Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate',
     'Director', 'Marketing Analyst', 'Product Manager', 'Sales Manager',
     'Marketing Coordinator', 'Senior Scientist', 'Software Developer',
     'HR Manager', 'Financial Analyst', 'Project Manager', 'Customer Service Rep',
     'Operations Manager', 'Marketing Manager', 'Senior Engineer',
     'Data Entry Clerk', 'Sales Director', 'Business Analyst', 'VP of Operations',
     'IT Support', 'Recruiter', 'Financial Manager', 'Social Media Specialist',
     'Software Manager', 'Junior Developer', 'Senior Consultant',
     'Product Designer', 'CEO', 'Accountant', 'Data Scientist',
     'Marketing Specialist', 'Technical Writer', 'HR Generalist',
     'Project Engineer', 'Customer Success Rep', 'Sales Executive', 'UX Designer',
     'Operations Director', 'Network Engineer', 'Administrative Assistant',
     'Strategy Consultant', 'Copywriter', 'Account Manager',
     'Director of Marketing', 'Help Desk Analyst', 'Customer Service Manager',
     'Business Intelligence Analyst', 'Event Coordinator', 'VP of Finance',
     'Graphic Designer', 'UX Researcher', 'Social Media Manager',
     'Director of Operations', 'Senior Data Scientist', 'Junior Accountant',
     'Digital Marketing Manager', 'IT Manager',
     'Customer Service Representative', 'Business Development Manager',
     'Senior Financial Analyst', 'Web Developer', 'Research Director',
     'Technical Support Specialist', 'Creative Director',
     'Senior Software Engineer', 'Human Resources Director',
     'Content Marketing Manager', 'Technical Recruiter', 'Sales Representative',
     'Chief Technology Officer', 'Junior Designer', 'Financial Advisor',
     'Junior Account Manager', 'Senior Project Manager', 'Principal Scientist',
     'Supply Chain Manager', 'Senior Marketing Manager', 'Training Specialist',
     'Research Scientist', 'Junior Software Developer',
     'Public Relations Manager', 'Operations Analyst',
     'Product Marketing Manager', 'Senior HR Manager', 'Junior Web Developer',
     'Senior Project Coordinator', 'Chief Data Officer',
     'Digital Content Producer', 'IT Support Specialist',
     'Senior Marketing Analyst', 'Customer Success Manager',
     'Senior Graphic Designer', 'Software Project Manager',
     'Supply Chain Analyst', 'Senior Business Analyst',
     'Junior Marketing Analyst', 'Office Manager', 'Principal Engineer',
     'Junior HR Generalist', 'Senior Product Manager',
     'Junior Operations Analyst', 'Senior HR Generalist',
     'Sales Operations Manager', 'Senior Software Developer',
     'Junior Web Designer', 'Senior Training Specialist',
     'Senior Research Scientist', 'Junior Sales Representative',
     'Junior Marketing Manager', 'Junior Data Analyst',
     'Senior Product Marketing Manager', 'Junior Business Analyst',
     'Senior Sales Manager', 'Junior Marketing Specialist',
     'Junior Project Manager', 'Senior Accountant', 'Director of Sales',
     'Junior Recruiter', 'Senior Business Development Manager',
     'Senior Product Designer', 'Junior Customer Support Specialist',
     'Senior IT Support Specialist', 'Junior Financial Analyst',
     'Senior Operations Manager', 'Director of Human Resources',
     'Junior Software Engineer', 'Senior Sales Representative',
     'Director of Product Management', 'Junior Copywriter',
     'Senior Marketing Coordinator', 'Senior Human Resources Manager',
     'Junior Business Development Associate', 'Senior Account Manager',
     'Senior Researcher', 'Junior HR Coordinator', 'Director of Finance',
     'Junior Marketing Coordinator', 'Junior Data Scientist',
     'Senior Operations Analyst', 'Senior Human Resources Coordinator',
     'Senior UX Designer', 'Junior Product Manager',
     'Senior Marketing Specialist', 'Senior IT Project Manager',
     'Senior Quality Assurance Analyst', 'Director of Sales and Marketing',
     'Senior Account Executive', 'Director of Business Development',
     'Junior Social Media Manager', 'Senior Human Resources Specialist',
     'Senior Data Analyst', 'Director of Human Capital',
     'Junior Advertising Coordinator', 'Junior UX Designer',
     'Senior Marketing Director', 'Senior IT Consultant',
     'Senior Financial Advisor', 'Junior Business Operations Analyst',
     'Junior Social Media Specialist', 'Senior Product Development Manager',
     'Junior Operations Manager', 'Senior Software Architect',
     'Junior Research Scientist', 'Senior Financial Manager',
     'Senior HR Specialist', 'Senior Data Engineer',
     'Junior Operations Coordinator', 'Director of HR',
     'Senior Operations Coordinator', 'Junior Financial Advisor',
     'Director of Engineering'])
    years_experience = st.number_input("Years of Experience", min_value=0, max_value=40, step=1)

    # Tombol untuk prediksi
    if st.button("Predict Salary"): # kode di dalam blok if akan dijalankan
        # Preprocessing input
        input_data = preprocess_input(age, gender, education, job_title, years_experience) #predict, diolah oleh fungsi preprocess_input.

        if input_data is not None: #memeriksa apakah data yang sudah diolah tadi valid.
            # Prediksi
            predicted_salary = model.predict(input_data)[0] #memprediksi gaji berdasarkan data yang sudah diolah.

            # Tampilkan hasil prediksi
            st.success(f"Predicted Salary: ${predicted_salary:,.2f}")
        else: 
            st.error("Input tidak dapat diproses. Mohon periksa kembali.") 

if __name__ == '__main__':
    main() #mengontrol kapa, bagaimana kode sebuah Python dijalankan