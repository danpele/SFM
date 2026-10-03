// ============================================================
// SFM site configuration (shared by index.html and index_ro.html)
// GOOGLE_CLIENT_ID: OAuth client (Web) of the Google Cloud project "MFM Quiz Login",
//   shared with the MFM site (same origin https://danpele.github.io);
//   the quizzes require "Sign in with Google" with an ASE account (@ase.ro / @stud.ase.ro).
// QUIZ_SCORES_URL: Google Apps Script web app that verifies the Google token and stores
//   the score in the Google Sheet "SFM 2026/2027 - Scoruri quiz" (Apps Script project "SFM Quiz Scores 2026-2027").
// ATTENDANCE_FORM_URL / ATTENDANCE_QR_URL: Google Form "SFM 2026/2027 - Prezență / Attendance" and the QR page
//   of the Apps Script project "SFM Prezenta 2026-2027" (ASE accounts only).
// ============================================================
window.SFM_CONFIG = {
    GOOGLE_CLIENT_ID: '1095360272769-rhjjncfor0gumhev6a0l6tnnrnmdrnna.apps.googleusercontent.com',
    ATTENDANCE_FORM_URL: 'https://forms.gle/bvES1ZVDmLZKPp3e9',
    ATTENDANCE_QR_URL: 'https://script.google.com/a/macros/ase.ro/s/AKfycbxKkgX5Jnq2e-y22xUjgI6fwMM4_TSIb6H4AeJ7vLABG-geBkyi6frM9CYKtrIA2O1F/exec',
    // The QR links appear on the site only after one of these accounts signs in with Google
    // TODO: replace the placeholder with the seminar instructor's ASE address
    INSTRUCTORS: ['danpele@ase.ro', 'YOUR_SEMINAR_INSTRUCTOR_EMAIL'],
    QUIZ_SCORES_URL: 'https://script.google.com/macros/s/AKfycbxr15dj2J-0D4unaq0l2IRroR1CU_fC1GVRWH_sOTtOvt5EWkA_I7TkkqdAEaztwe-T/exec'
};
