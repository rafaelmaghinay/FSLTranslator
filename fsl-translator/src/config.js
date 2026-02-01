const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const SERVER_BASE = API_URL;
export const WS_URL = API_URL.replace(/^http/, 'ws') + '/api/live';