import * as d3 from 'd3'
import { D3Sel } from "./etc/Util"
import { Histogram } from './vis/Histogram'
import { SimpleEventHandler } from './etc/SimpleEventHandler'
import { API } from './api/mainApi'
import { State, URLParameters } from './state'
import { caseStudyOptions, methodOptions, sortByOptions, predictionFnOptions, scoreFnOptions, labelFilterOptions, caseOptions, caseValues } from './etc/selectionOptions'
import { SaliencyTextViz } from "./vis/SaliencyTextRow"
import { SaliencyTexts } from "./vis/SaliencyTexts"

/**
 * Render static elements needed for interface
 */
function init(base: D3Sel) {
    const html = `
    <!--  Filter Controls  -->
    <div class="controls container-md cont-nav">
        <div class="form-row">
            <div class="col-sm-2">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="case-study-select">Data</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_case-study-select">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>

            <div class="col-sm-2">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="saliency-method-select">Saliency</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_saliency-method-select">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>

            <div class="col-sm-2">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="scorefn-select">Score</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_scorefn-select">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>

            <div class="col-sm-2">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="sort-by-select">Sort</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_sort-by-select">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>

            <div class="col-sm-2">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="label-filter">Label</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_label-filter">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>

            <div class="col-sm-2">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="prediction-filter">Prediction</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_prediction-filter">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>

        </div>
    </div>

    <!--  Results  -->
    <div class="ID_main">
        <div class="ID_sidebar">
            <div class="ID_number-of-results">Filtering to x of Y</div>
            <!--  Cases  -->
            <div class="cases">
                <div class="input-group input-group-sm mb-3">
                    <div class="input-group-prepend">
                        <label class="input-group-text" for="case-filter">Case</label>
                    </div>
                    <select class="custom-select custom-select-sm ID_cases">
                        <!-- Fill in from data in TS now -->
                    </select>
                </div>
            </div>
            <div class="ID_case-description"></div>
        </div>
        <div class="ID_mainpage">
            <div class="ID_results-panel"></div>
        </div>

    </div>
    `

    base.html(html)
}

/**
 * Main functionality in the below function
 */
export function main(el: Element, ignoreUrl: boolean = false, stateParams: Partial<URLParameters> = {}, freezeParams: boolean = false) {
    const base = d3.select(el)

    const eventHandler = new SimpleEventHandler(el)
    const api = new API()
    const state = new State(ignoreUrl, stateParams, freezeParams)

    init(base)
    const selectors = {
        body: d3.select('body'),
        main: base.select('.ID_main'),
        navBar: base.select('.controls'),
        mainPage: base.select('.ID_mainpage'),
        resultsPanel: base.select('.ID_results-panel'),
        sidebar: base.select('.ID_sidebar'),
        caseStudy: base.select('.ID_case-study-select'),
        caseStudyListOptions: base.select('.ID_case-study-select').selectAll('option')
            .data(caseStudyOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        method: base.select('.ID_saliency-method-select'),
        methodListOptions: base.select('.ID_saliency-method-select').selectAll('option')
            .data(methodOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        scoreFn: base.select('.ID_scorefn-select'),
        scoreFnListOptions: base.select('.ID_scorefn-select').selectAll('option')
            .data(scoreFnOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        sortBy: base.select('.ID_sort-by-select'),
        sortByListOptions: base.select('.ID_sort-by-select').selectAll('option')
            .data(sortByOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        predictionFn: base.select('.ID_prediction-filter'),
        predictionFnListOptions: base.select('.ID_prediction-filter').selectAll('option')
            .data(predictionFnOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        labelFilter: base.select('.ID_label-filter'),
        labelFilterListOptions: base.select('.ID_label-filter').selectAll('option')
            .data(labelFilterOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        numberOfResults: base.select('.ID_number-of-results'),
        caseFilter: base.select('.ID_cases'),
        caseListOptions: base.select('.ID_cases').selectAll('option')
            .data(caseOptions)
            .join('option')
            .attr('value', option => option.value)
            .text(option => option.name),
        caseDescription: base.select('.ID_case-description').classed("description", true)
    }

    const vizs = {
        IouHistogram: new Histogram(<HTMLElement>selectors.sidebar.node(), 'IoU', eventHandler),
        ECHistogram: new Histogram(<HTMLElement>selectors.sidebar.node(), 'Saliency Coverage', eventHandler),
        GTCHistogram: new Histogram(<HTMLElement>selectors.sidebar.node(), 'Ground Truth Coverage', eventHandler),
        results: new SaliencyTexts(<HTMLElement>selectors.resultsPanel.node(), eventHandler)
    }

    const eventHelpers = {
        /**
        * Update the results panel.
        * @param {State} state - the current state of the application.
        */
        updateResults: (state: State) => {
            api.getResultIDs(state.caseStudy(), state.method(), state.sortBy(), state.predictionFn(), state.scoreFn(), state.labelFilter(),
                             state.iouFilter(), state.explanationFilter(), state.groundTruthFilter()).then(IDs => {
                // Set the number of results
                state.resultCount(IDs.length)
                eventHelpers.updateResultCount(state)

                // Update results
                vizs.results.update(IDs)
            })
        },

        /**
        * Update the results panel, histogram, and confusion matrix.
        * @param {State} state - the current state of the application.
        */
        updatePage: (state: State) => {
            // Update histograms using all results
            const allImageIDs = api.getResultIDs(state.caseStudy(), state.method(), state.sortBy(), 'all', state.scoreFn(),
                '', [0, 1], [0, 1], [0, 1])
            selectors.body.style('cursor', 'progress')
            allImageIDs.then(IDs => {
                // Update number of results
                state.totalResultCount(IDs.length)

                // Update histograms
                api.binScores(state.caseStudy(), state.method(), IDs, 'iou').then(bins => {
                    vizs.IouHistogram.update({bins: bins, brushRange: state.iouFilter()})
                })
                api.binScores(state.caseStudy(), state.method(), IDs, 'explanation_coverage').then(bins => {
                    vizs.ECHistogram.update({bins: bins, brushRange: state.explanationFilter()})
                })
                api.binScores(state.caseStudy(), state.method(), IDs, 'ground_truth_coverage').then(bins => {
                    vizs.GTCHistogram.update({bins: bins, brushRange: state.groundTruthFilter()})
                })
                selectors.body.style('cursor', 'default')
            })

            // Update image panel
            vizs.results.clear()
            const imageIDs = api.getResultIDs(state.caseStudy(), state.method(), state.sortBy(), state.predictionFn(), state.scoreFn(),
                state.labelFilter(), state.iouFilter(), state.explanationFilter(), state.groundTruthFilter())
            selectors.body.style('cursor', 'progress')
            imageIDs.then(IDs => {
                // Set the number of results
                state.resultCount(IDs.length)
                eventHelpers.updateResultCount(state)

                // Set images
                vizs.results.update(IDs)
                selectors.body.style('cursor', 'default')
            })
        },

        /**
        * Update the label drop down values.
        * @param {State} state - the current state of the application.
        */
        updateLabels: (state: State) => {
            api.getLabels(state.caseStudy(), state.method()).then(labels => {
                const labelValues = labels.slice().sort(d3.ascending);
                labels.sort(d3.ascending).splice.apply(labels, [0, 0 as string | number].concat(labelFilterOptions.map(option => option.name)));
                labelValues.splice.apply(labelValues, [0, 0 as string | number].concat(labelFilterOptions.map(option => option.value)));
                selectors.labelFilter.selectAll('option')
                    .data(labels)
                    .join('option')
                    .attr('value', (label, i) => labelValues[i])
                    .attr('disabled', state.isFrozen('labelFilter'))
                    .text(label => label)
                selectors.labelFilter.property('value', state.labelFilter())
            })
        },

        /**
        * Update the prediction drop down values.
        * @param {State} state - the current state of the application.
        */
        updatePredictions: (state: State) => {
            api.getPredictions(state.caseStudy(), state.method()).then(predictions => {
                const predictionValues = predictions.slice().sort(d3.ascending);
                predictions.sort(d3.ascending).splice.apply(predictions, [0, 0 as string | number].concat(predictionFnOptions.map(option => option.name)));
                predictionValues.splice.apply(predictionValues, [0, 0 as string | number].concat(predictionFnOptions.map(option => option.value)));
                console.log("Prediction Values: ", predictions);
                selectors.predictionFn.selectAll('option')
                    .data(predictions)
                    .join('option')
                    .attr('value', (prediction, i) => predictionValues[i])
                    .attr('disabled', state.isFrozen('predictionFn'))
                    .text(prediction => {
                        return prediction
                    })
                selectors.predictionFn.property('value', state.predictionFn())
            })
        },

        /**
        * Update the case if the filters have changed.
        * @param {State} state - the current state of the application.
        */
         updateCase: (state: State) => {
            const currentCase = selectors.caseFilter.property('value')
            const currentPredictionFilter = selectors.predictionFn.property('value')
            if (currentCase != 'default') { 
                // const cf = caseValues[currentCase]
                // state.scoreFn(cf['selectedScore'])
                // state.sortBy(cf['sortBy'])
                const caseScores = caseValues[currentCase]['scores']
                if ( currentPredictionFilter != caseValues[currentCase]['prediction'] ||
                state.iouFilter()[0] != caseScores['iou'][0] || 
                state.iouFilter()[1] != caseScores['iou'][1] || 
                state.groundTruthFilter()[0] != caseScores['ground_truth_coverage'][0] ||
                state.groundTruthFilter()[1] != caseScores['ground_truth_coverage'][1] ||
                state.explanationFilter()[0] != caseScores['explanation_coverage'][0] ||
                state.explanationFilter()[1] != caseScores['explanation_coverage'][1] ) {
                    selectors.caseFilter.property('value', 'default')
                    selectors.caseDescription.text(caseValues['default']['description'])
                }
            }
        },

        /**
        * Update the result count.
        * @param {State} state - the current state of the application.
        */
         updateResultCount: (state: State) => {
            selectors.numberOfResults.text('Filtering to ' + state.resultCount() + ' of ' + state.totalResultCount() + ' results')
        },
    }

    /**
     * Initialize the application from the state.
     * @param {State} state - the state of the application.
     */
    async function initializeFromState(state: State) {
        // Fill in label and prediction options
        eventHelpers.updateLabels(state)
        eventHelpers.updatePredictions(state)

        // Set frontend via state parameters
        selectors.caseStudy.property('value', state.caseStudy())
        selectors.method.property('value', state.method())
        selectors.sortBy.property('value', state.sortBy())
        selectors.scoreFn.property('value', state.scoreFn())

        // Get data from state parameters
        eventHelpers.updatePage(state)
    }

    initializeFromState(state)

    selectors.caseStudy.on('change', () => {
        /* When the case study changes, update the page with the new data. */
        console.log("CaseStudy Changing");
        const caseStudy = selectors.caseStudy.property('value')
        state.caseStudy(caseStudy)
        state.labelFilter('')
        eventHelpers.updateLabels(state)
        state.predictionFn('all')
        eventHelpers.updatePredictions(state)
        eventHelpers.updatePage(state)
    });

    selectors.method.on('change', () => {
        /* When the method changes, update the page with the new data. */
        const method = selectors.method.property('value')
        state.method(method)
        state.labelFilter('')
        eventHelpers.updateLabels(state)
        state.predictionFn('all')
        eventHelpers.updatePredictions(state)
        eventHelpers.updatePage(state)
    });

    selectors.sortBy.on('change', () => {
        /* When the sort by value changes, update the results panel. */
        const sortByValue = selectors.sortBy.property('value')
        state.sortBy(sortByValue)
        eventHelpers.updateResults(state)
    });

    selectors.predictionFn.on('change', () => {
        /* When the prediction function changes, update the page. */
        const predictionValue = selectors.predictionFn.property('value')
        state.predictionFn(predictionValue)
        eventHelpers.updatePage(state)
    });

    selectors.scoreFn.on('change', () => {
        /* When the score function changes, update the page. */
        const scoreValue = selectors.scoreFn.property('value')
        state.scoreFn(scoreValue)
        eventHelpers.updatePage(state)
    });

    selectors.labelFilter.on('change', () => {
        /* When the label filter changes, update the page. */
        const labelFilter = selectors.labelFilter.property('value')
        state.labelFilter(labelFilter)
        eventHelpers.updatePage(state)
    });

    selectors.caseFilter.on('change', () => {
        /* When case changes, update the page. */
        const caseFilter = selectors.caseFilter.property('value')
        if (caseFilter) { 
            const cf = caseValues[caseFilter]
            const caseFilterScores = cf['scores']
            state.scoreFn(cf['selectedScore'])
            selectors.scoreFn.property('value', state.scoreFn())
            state.sortBy(cf['sortBy'])
            selectors.sortBy.property('value', state.sortBy())
            state.iouFilter(caseFilterScores['iou'][0], caseFilterScores['iou'][1])
            state.groundTruthFilter(caseFilterScores['ground_truth_coverage'][0], caseFilterScores['ground_truth_coverage'][1])
            state.explanationFilter(caseFilterScores['explanation_coverage'][0], caseFilterScores['explanation_coverage'][1])
            state.predictionFn(caseValues[caseFilter]['prediction'])
            selectors.caseDescription.text(caseValues[caseFilter]['description'])
            eventHelpers.updatePredictions(state)
            eventHelpers.updatePage(state)
        }
    }) 

    eventHandler.bind(SaliencyTexts.events.onScreen, ({ el, id, caller }) => {
        /* Lazy load the saliency results. */
        const row = new SaliencyTextViz(el, eventHandler)
        api.getResult(state.caseStudy(), state.method(), id, state.scoreFn()).then(salTxt => {
            row.update(salTxt)
        })
    });

    eventHandler.bind(Histogram.events.onBrush, ({ minScore, maxScore, score, caller }) => {
        /* Filter scores */
        minScore = Math.round((minScore + Number.EPSILON) * 100) / 100
        maxScore = Math.round((maxScore + Number.EPSILON) * 100) / 100
        if (score == 'IoU') { 
            state.iouFilter(minScore, maxScore)
        } else if (score == 'Saliency Coverage') { 
            state.explanationFilter(minScore, maxScore)
        } else if (score == 'Ground Truth Coverage') { 
            state.groundTruthFilter(minScore, maxScore)
        }
        eventHelpers.updateResults(state)

        // Reset case if necessary 
        eventHelpers.updateCase(state)
    });

}