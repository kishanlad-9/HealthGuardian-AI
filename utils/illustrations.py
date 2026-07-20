"""Healthcare-themed SVG illustration for the login page's right panel.
Kept separate from the page file since it's a large string constant."""

LOGIN_ILLUSTRATION_SVG = """
<div class="hg-illustration-panel">
  <div class="hg-float-badge hg-float-1">
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
      <circle cx="17" cy="17" r="17" fill="#4FD1AE"/>
      <path d="M17 9v16M9 17h16" stroke="white" stroke-width="3" stroke-linecap="round"/>
    </svg>
  </div>
  <div class="hg-float-badge hg-float-2">
    <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
      <circle cx="15" cy="15" r="15" fill="#FFFFFF"/>
      <path d="M9 15.5l3 3 9-9" stroke="#5B5FEF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div class="hg-float-badge hg-float-3">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
      <circle cx="13" cy="13" r="13" fill="#E3FBF3"/>
      <path d="M13 7a6 6 0 106 6 6 6 0 00-6-6z" stroke="#0E9F79" stroke-width="1.6" fill="none"/>
    </svg>
  </div>

  <svg class="hg-center-badge" width="220" height="220" viewBox="0 0 220 220" fill="none">
    <circle cx="110" cy="110" r="108" fill="rgba(255,255,255,0.08)"/>
    <circle cx="110" cy="110" r="86" fill="rgba(255,255,255,0.10)"/>
    <circle cx="110" cy="110" r="64" fill="#FFFFFF"/>
    <path d="M40 110 h30 l10 -26 l16 52 l12 -34 l8 8 h74"
          stroke="#5B5FEF" stroke-width="4" fill="none"
          stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="110" cy="110" r="6" fill="#4FD1AE"/>
  </svg>

  <p class="hg-illustration-tagline">Predict early. Understand clearly.<br/>Take charge of your health.</p>
</div>

<style>
.hg-illustration-panel {
    position: relative;
    height: 100%;
    min-height: 460px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(160deg, var(--brand-navy) 0%, var(--brand-indigo) 100%);
    border-radius: 20px;
    overflow: hidden;
    padding: 2rem;
}
.hg-center-badge {
    animation: hg-pulse 3s ease-in-out infinite;
}
.hg-illustration-tagline {
    color: #E8EAF7;
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    margin-top: 1.8rem;
    font-size: 1.05rem;
    line-height: 1.5;
}
.hg-float-badge {
    position: absolute;
    filter: drop-shadow(0 6px 14px rgba(0,0,0,0.15));
}
.hg-float-1 { top: 14%; left: 14%; animation: hg-float 3.5s ease-in-out infinite; }
.hg-float-2 { top: 20%; right: 12%; animation: hg-float 4s ease-in-out infinite 0.5s; }
.hg-float-3 { bottom: 16%; left: 20%; animation: hg-float 3s ease-in-out infinite 1s; }
</style>
"""
