document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.main-nav a');
    const pageMeta = document.body.dataset.page;

    const pageMap = {
        home: '/',
        training: '/training-console',
        pipeline: '/pipeline-explorer',
        evaluation: '/evaluation-report',
        predict: '/live-inference'
    };

    if (pageMeta && pageMap[pageMeta]) {
        const activeLink = document.querySelector(`.main-nav a[href="${pageMap[pageMeta]}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
});