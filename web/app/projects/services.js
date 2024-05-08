import { destroy, get, patch, post } from "@/services/http";

const localeService = {
  get: async (id) => {
    const url = `/api/locales/${id}`;
    const errMsg = `Failed to fetch locale ${id}.`;
    return await get(url, errMsg);
  },
  list: async (searchPhrase) => {
    if (!searchPhrase) return [];
    const url = "/api/locales";
    const errMsg = `Locale search with query \"${searchPhrase}\" failed.`;
    let requestBody = {
      searchPhrase: searchPhrase,
      limit: parseInt(process.env.NEXT_PUBLIC_NUM_SEARCH_RESULTS),
    };
    return await post(url, requestBody, errMsg);
  },
};

const projectService = {
  create: async (project) => {
    const url = "/api/projects";
    const errMsg = `Failed to create project: ${JSON.stringify(project)}.`;
    return await post(url, project, errMsg);
  },
  delete: async (id) => {
    const url = `/api/projects/${id}`;
    const errMsg = `Failed to delete project ${id}.`;
    return await destroy(url, errMsg);
  },
  get: async (id) => {
    const url = `/api/projects/${id}`;
    const errMsg = `Failed to fetch project ${id}.`;
    return await get(url, errMsg);
  },
  list: async (limit, offset) => {
    let url = `/api/projects?limit=${limit}&offset=${offset}`;
    let errMsg = `Failed to return projects.`;
    return await get(url, errMsg);
  },
  update: async (id, project) => {
    const url = `/api/projects/${id}`;
    const errMsg = `Failed to update project: ${JSON.stringify(project)}.`;
    return await patch(url, project, errMsg);
  },
};

const jobService = {
  queue: async (projectId) => {
    let url = process.env.LOCAL_JOB_SERVICE_URL;
    let payload = { projectId: projectId };
    let errMsg = `Failed to queue job for model ${projectId}.`;
    return await post(url, payload, errMsg);
  },
  getStatus: async (jobId) => {
    let url = `${process.env.LOCAL_JOB_SERVICE_URL}/${jobId}`;
    let errMsg = `Failed to fetch status for job ${jobId}.`;
    return await get(url, errMsg);
  },
};

export { localeService, projectService, jobService };
