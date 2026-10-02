// ============================================================
// SFM site configuration (shared by index.html and index_ro.html)
// GOOGLE_CLIENT_ID: OAuth client (Web) of the Google Cloud project "MFM Quiz Login",
//   shared with the MFM site (same origin https://danpele.github.io);
//   the quizzes require "Sign in with Google" with an ASE account (@ase.ro / @stud.ase.ro).
// QUIZ_SCORES_URL: Google Apps Script web app that verifies the Google token and stores
//   the score in a Google Sheet. TODO: deploy an SFM copy of the MFM script (own Sheet)
//   and paste its /exec URL here. While it starts with 'YOUR_', scores are not sent.
// ATTENDANCE_FORM_URL / ATTENDANCE_QR_URL: TODO for SFM; hidden while they start with 'YOUR_'.
// ============================================================
window.SFM_CONFIG = {
    GOOGLE_CLIENT_ID: '1095360272769-rhjjncfor0gumhev6a0l6tnnrnmdrnna.apps.googleusercontent.com',
    ATTENDANCE_FORM_URL: 'YOUR_SFM_ATTENDANCE_FORM_URL',
    ATTENDANCE_QR_URL: 'YOUR_SFM_ATTENDANCE_QR_URL',
    // The QR links appear on the site only after one of these accounts signs in with Google
    // TODO: replace the placeholder with the seminar instructor's ASE address
    INSTRUCTORS: ['danpele@ase.ro', 'YOUR_SEMINAR_INSTRUCTOR_EMAIL'],
    QUIZ_SCORES_URL: 'YOUR_SFM_QUIZ_SCORES_URL'
};
