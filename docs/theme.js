const themeToggle = document.getElementById('theme-toggle');
const themeIconLight = document.getElementById('theme-icon-light');
const themeIconDark = document.getElementById('theme-icon-dark');
const body = document.body;

let currentTheme = 'light';

/**
 * Sets the application theme and saves the preference.
 * @param {string} theme - The theme to set ('light', 'dark', 'system').
 */
function setTheme(theme) {
    let themeToApply = theme;
    if (theme === 'system') {
        themeToApply = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    currentTheme = themeToApply;
    body.setAttribute('data-theme', currentTheme);
    localStorage.setItem('arxiv_theme', theme); // Save the preference (light, dark, or system)
    updateThemeIcons();
    console.log(`Theme set to: ${theme} (applied: ${currentTheme})`);
}

/**
 * Toggles between light and dark themes.
 */
function toggleTheme() {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    // We can show a toast from the main app if needed
}

function updateThemeIcons() {
    if (currentTheme === 'dark') {
        themeIconLight.classList.add('hidden');
        themeIconDark.classList.remove('hidden');
    } else {
        themeIconLight.classList.remove('hidden');
        themeIconDark.classList.add('hidden');
    }
}

/**
 * Initializes the theme based on saved preference or system settings.
 */
export function initializeTheme() {
    const savedTheme = localStorage.getItem('arxiv_theme') || 'system';
    setTheme(savedTheme);
    themeToggle.addEventListener('click', toggleTheme);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => setTheme(localStorage.getItem('arxiv_theme') || 'system'));
}