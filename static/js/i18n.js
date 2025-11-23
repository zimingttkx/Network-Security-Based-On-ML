// 多语言支持系统
const translations = {
    'zh-CN': {
        // 导航栏
        'nav.home': '首页',
        'nav.predict': '威胁预测',
        'nav.train': '模型训练',
        'nav.tutorial': '使用教程',
        'nav.api': 'API文档',

        // 首页
        'home.title': '网络安全威胁检测系统',
        'home.subtitle': '基于机器学习的实时网络流量异常检测与预警平台',
        'home.feature1.title': '智能检测',
        'home.feature1.desc': '使用先进的机器学习算法，准确识别网络威胁',
        'home.feature2.title': '实时分析',
        'home.feature2.desc': '毫秒级响应，实时分析网络流量数据',
        'home.feature3.title': '易于使用',
        'home.feature3.desc': '直观的界面，即使是新手也能快速上手',
        'home.feature4.title': '高可用性',
        'home.feature4.desc': '企业级架构，支持大规模部署',
        'home.getStarted': '开始使用',
        'home.learnMore': '了解更多',

        // 预测页面
        'predict.title': '威胁预测',
        'predict.upload.title': '上传CSV文件',
        'predict.upload.desc': '选择包含网络流量数据的CSV文件进行批量预测',
        'predict.upload.button': '选择文件',
        'predict.upload.submit': '开始预测',
        'predict.json.title': '手动输入数据',
        'predict.json.desc': '输入网络流量特征数据（JSON格式）',
        'predict.json.placeholder': '输入JSON格式的数据，例如：\n[[1.0, 2.0, 3.0, ...]]',
        'predict.json.submit': '立即预测',
        'predict.result.title': '预测结果',
        'predict.result.threat': '威胁级别',
        'predict.result.safe': '安全',
        'predict.result.dangerous': '危险',
        'predict.result.confidence': '置信度',

        // 训练页面
        'train.title': '模型训练',
        'train.desc': '训练新的机器学习模型以提高检测准确率',
        'train.status': '训练状态',
        'train.start': '开始训练',
        'train.stop': '停止训练',
        'train.progress': '训练进度',
        'train.metrics': '性能指标',
        'train.accuracy': '准确率',
        'train.precision': '精确率',
        'train.recall': '召回率',
        'train.f1': 'F1分数',
        'train.logs': '训练日志',

        // 教程页面
        'tutorial.title': '使用教程',
        'tutorial.welcome': '欢迎使用网络安全威胁检测系统',
        'tutorial.quick': '快速开始',
        'tutorial.step1': '步骤 1：准备数据',
        'tutorial.step1.desc': '准备您的网络流量数据文件（CSV格式）',
        'tutorial.step2': '步骤 2：上传预测',
        'tutorial.step2.desc': '在"威胁预测"页面上传文件或输入数据',
        'tutorial.step3': '步骤 3：查看结果',
        'tutorial.step3.desc': '系统会立即显示预测结果和威胁级别',
        'tutorial.faq': '常见问题',
        'tutorial.video': '视频教程',

        // 通用
        'common.loading': '加载中...',
        'common.error': '错误',
        'common.success': '成功',
        'common.cancel': '取消',
        'common.confirm': '确认',
        'common.close': '关闭',
        'common.download': '下载',
        'common.upload': '上传',
        'common.status': '状态',
        'common.healthy': '健康',
        'common.version': '版本',
    },

    'en-US': {
        // Navigation
        'nav.home': 'Home',
        'nav.predict': 'Threat Prediction',
        'nav.train': 'Model Training',
        'nav.tutorial': 'Tutorial',
        'nav.api': 'API Docs',

        // Home
        'home.title': 'Network Security Threat Detection System',
        'home.subtitle': 'Real-time network traffic anomaly detection and alert platform based on machine learning',
        'home.feature1.title': 'Intelligent Detection',
        'home.feature1.desc': 'Accurately identify network threats using advanced machine learning algorithms',
        'home.feature2.title': 'Real-time Analysis',
        'home.feature2.desc': 'Millisecond-level response for real-time network traffic analysis',
        'home.feature3.title': 'Easy to Use',
        'home.feature3.desc': 'Intuitive interface that even beginners can quickly master',
        'home.feature4.title': 'High Availability',
        'home.feature4.desc': 'Enterprise-grade architecture supporting large-scale deployment',
        'home.getStarted': 'Get Started',
        'home.learnMore': 'Learn More',

        // Predict
        'predict.title': 'Threat Prediction',
        'predict.upload.title': 'Upload CSV File',
        'predict.upload.desc': 'Select a CSV file containing network traffic data for batch prediction',
        'predict.upload.button': 'Choose File',
        'predict.upload.submit': 'Start Prediction',
        'predict.json.title': 'Manual Input',
        'predict.json.desc': 'Enter network traffic feature data (JSON format)',
        'predict.json.placeholder': 'Enter data in JSON format, e.g.:\n[[1.0, 2.0, 3.0, ...]]',
        'predict.json.submit': 'Predict Now',
        'predict.result.title': 'Prediction Results',
        'predict.result.threat': 'Threat Level',
        'predict.result.safe': 'Safe',
        'predict.result.dangerous': 'Dangerous',
        'predict.result.confidence': 'Confidence',

        // Train
        'train.title': 'Model Training',
        'train.desc': 'Train a new machine learning model to improve detection accuracy',
        'train.status': 'Training Status',
        'train.start': 'Start Training',
        'train.stop': 'Stop Training',
        'train.progress': 'Progress',
        'train.metrics': 'Performance Metrics',
        'train.accuracy': 'Accuracy',
        'train.precision': 'Precision',
        'train.recall': 'Recall',
        'train.f1': 'F1 Score',
        'train.logs': 'Training Logs',

        // Tutorial
        'tutorial.title': 'Tutorial',
        'tutorial.welcome': 'Welcome to Network Security Threat Detection System',
        'tutorial.quick': 'Quick Start',
        'tutorial.step1': 'Step 1: Prepare Data',
        'tutorial.step1.desc': 'Prepare your network traffic data file (CSV format)',
        'tutorial.step2': 'Step 2: Upload & Predict',
        'tutorial.step2.desc': 'Upload file or enter data on the "Threat Prediction" page',
        'tutorial.step3': 'Step 3: View Results',
        'tutorial.step3.desc': 'The system will immediately display prediction results and threat levels',
        'tutorial.faq': 'FAQ',
        'tutorial.video': 'Video Tutorial',

        // Common
        'common.loading': 'Loading...',
        'common.error': 'Error',
        'common.success': 'Success',
        'common.cancel': 'Cancel',
        'common.confirm': 'Confirm',
        'common.close': 'Close',
        'common.download': 'Download',
        'common.upload': 'Upload',
        'common.status': 'Status',
        'common.healthy': 'Healthy',
        'common.version': 'Version',
    },

    'ja-JP': {
        // ナビゲーション
        'nav.home': 'ホーム',
        'nav.predict': '脅威予測',
        'nav.train': 'モデル訓練',
        'nav.tutorial': 'チュートリアル',
        'nav.api': 'API文書',

        // ホーム
        'home.title': 'ネットワークセキュリティ脅威検出システム',
        'home.subtitle': '機械学習に基づくリアルタイムネットワークトラフィック異常検出および警告プラットフォーム',
        'home.feature1.title': 'インテリジェント検出',
        'home.feature1.desc': '高度な機械学習アルゴリズムを使用してネットワーク脅威を正確に識別',
        'home.feature2.title': 'リアルタイム分析',
        'home.feature2.desc': 'ミリ秒レベルの応答でリアルタイムネットワークトラフィック分析',
        'home.feature3.title': '使いやすい',
        'home.feature3.desc': '直感的なインターフェースで初心者でもすぐに使える',
        'home.feature4.title': '高可用性',
        'home.feature4.desc': '大規模展開をサポートするエンタープライズグレードアーキテクチャ',
        'home.getStarted': '始める',
        'home.learnMore': '詳細',

        // 共通
        'common.loading': '読み込み中...',
        'common.error': 'エラー',
        'common.success': '成功',
        'common.cancel': 'キャンセル',
        'common.confirm': '確認',
        'common.close': '閉じる',
    }
};

// 当前语言
let currentLang = localStorage.getItem('language') || 'zh-CN';

// 翻译函数
function t(key) {
    return translations[currentLang][key] || key;
}

// 切换语言
function setLanguage(lang) {
    if (translations[lang]) {
        currentLang = lang;
        localStorage.setItem('language', lang);
        updatePageTexts();
    }
}

// 更新页面文本
function updatePageTexts() {
    document.querySelectorAll('[data-i18n]').forEach(elem => {
        const key = elem.getAttribute('data-i18n');
        const translation = t(key);

        if (elem.tagName === 'INPUT' || elem.tagName === 'TEXTAREA') {
            elem.placeholder = translation;
        } else {
            elem.textContent = translation;
        }
    });

    // 更新语言选择器
    const langSelector = document.getElementById('languageSelector');
    if (langSelector) {
        langSelector.value = currentLang;
    }
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    updatePageTexts();
});
