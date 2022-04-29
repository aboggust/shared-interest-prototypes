import * as d3 from 'd3';
import { makeUrl, toPayload } from '../etc/apiHelpers'
import { URLHandler } from '../etc/URLHandler';
import { SaliencyImg, Bins, ConfusionMatrixI } from '../types';


const baseurl = URLHandler.basicURL()


export class API {

    constructor(private baseURL: string = null) {
        if (this.baseURL == null) {
            this.baseURL = baseurl + '/api';
        }
    }


    /**
     * Get the saliency image objects for all imageIDs.
     *
     * @param {string} caseStudy - the name of the case study
     * @param {string} method - the name of the saliency method
     * @param {string[]} imageIDs - a list of string image ids
     * @param {string} scoreFn - the score function name
     * @return {Promise<SaliencyImg[]>} a list of SaliencyImg for the imageIDs in the caseStudy
     */
    getSaliencyImages(caseStudy: string, method: string, imageIDs: string[], scoreFn: string): Promise<SaliencyImg[]> {
        const imagesToSend = {
            case_study: caseStudy,
            method: method,
            image_ids: imageIDs,
            score_fn: scoreFn
        }
        const url = makeUrl(this.baseURL + "/get-saliency-images")
        const payload = toPayload(imagesToSend)
        return d3.json(url, payload)
    }

    /**
     * Get a saliency image objects for a requested imageID.
     *
     * @param {string} caseStudy - the name of the case study
     * @param {string} method - the name of the saliency method
     * @param {string[]} imageID - a list of string image ids
     * @param {string} scoreFn - the score function name
     * @return {Promise<SaliencyImg>} a SaliencyImg object for the imageID in the caseStudy.
     */
    getSaliencyImage(caseStudy: string, method: string, imageID: string, scoreFn: string): Promise<SaliencyImg> {
        const imagesToSend = {
            case_study: caseStudy,
            method: method,
            image_id: imageID,
            score_fn: scoreFn
        }
        const url = makeUrl(this.baseURL + "/get-saliency-image", imagesToSend)
        return d3.json(url)
    }


    /**
     * Get the imageIDs for the images fitting the filter parameters.
     *
     * @param {string} caseStudy - the name of the case study
     * @param {string} method - the name of the saliency method
     * @param {number} sortBy - 1 if sort ascending, -1 if sort descending
     * @param {string} predictionFn - the name of the prediction filter
     * @param {string} scoreFn - the score function name
     * @param {string} labelFilter - the name of the label filter
     * @param {string} iouFilter - min and max iou values
     * @param {string} groundTruthFilter - min and max ground truth coverage values
     * @param {string} explanationFilter - min and max explanation coverage values
     * @return {Promise<string[]>} a list of imageIDs
     */
    getImages(caseStudy: string, method: string, sortBy: number, predictionFn: string, scoreFn: string,
              labelFilter: string, iouFilter: number[], groundTruthFilter: number[],
              explanationFilter: number[]): Promise<string[]> {
        const toSend = {
            case_study: caseStudy,
            method: method,
            sort_by: sortBy,
            prediction_fn: predictionFn,
            score_fn: scoreFn,
            label_filter: labelFilter,
            iou_min: iouFilter[0],
            iou_max: iouFilter[1],
            ec_min: explanationFilter[0],
            ec_max: explanationFilter[1],
            gtc_min: groundTruthFilter[0],
            gtc_max: groundTruthFilter[1],
        }
        const url = makeUrl(this.baseURL + "/get-images", toSend)
        return d3.json(url)
    }

    /**
     * Get all dataset predictions.
     *
     * @param {string} caseStudy - the name of the case study
     * @return {Promise<string[]>} a list of all model predictions for caseStudy
     */
    getPredictions(caseStudy: string): Promise<string[]> {
        const toSend = {
            case_study: caseStudy
        }
        const url = makeUrl(this.baseURL + "/get-predictions", toSend)
        return d3.json(url)
    }

    /**
     * Get all dataset labels.
     *
     * @param {string} caseStudy - the name of the case study
     * @return {Promise<string[]>} a list of all image labels for caseStudy
     */
    getLabels(caseStudy: string): Promise<string[]> {
        const toSend = {
            case_study: caseStudy
        }
        const url = makeUrl(this.baseURL + "/get-labels", toSend)
        return d3.json(url)
    }

    /**
     * Get the histogram bins for the scoreFn scores of the imageIDs.
     *
     * @param {string} caseStudy - the name of the case study
     * @param {string} method - the name of the saliency method
     * @param {string[]} imageIDs - a list of string image ids
     * @param {string} scoreFn - the score function name
     * @return {Promise<Bins[]>} a list of Bins for the binned scores of the image IDs
     */
    binScores(caseStudy: string, method: string, imageIDs: string[], scoreFn: string): Promise<Bins[]> {
        const imagesToSend = {
            case_study: caseStudy,
            image_ids: imageIDs,
            score_fn: scoreFn
        }
        const url = makeUrl(this.baseURL + "/bin-scores")
        const payload = toPayload(imagesToSend)
        return d3.json(url, payload)
    }


};
