// Custom JS to hide the Readme button in the header dynamically
const hideReadme = () => {
    const elements = document.querySelectorAll('button, a, span, p, div');
    for (const el of elements) {
        if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3 && el.textContent.trim().toLowerCase() === 'readme') {
            const buttonOrLink = el.closest('a') || el.closest('button') || el;
            if (buttonOrLink && buttonOrLink.style.display !== 'none') {
                buttonOrLink.style.setProperty('display', 'none', 'important');
                console.log('[CustomJS] Hid Readme button');
            }
        }
    }
};

// Run periodically to handle dynamic renders in React UI
setInterval(hideReadme, 150);
