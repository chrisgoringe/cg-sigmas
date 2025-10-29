import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const text_widget_name = "display_text_widget$$"
const message_id = "cg_sigmas_display_text"

function get_text_widget(node) {
    var w = node.widgets?.find((w) => w.name === text_widget_name);
    if (!w) {
		w = ComfyWidgets["STRING"](node, text_widget_name, ["STRING", { multiline: true }], app).widget;
		w.inputEl.readOnly = true;
		w.inputEl.style.opacity = 0.6;
		w.inputEl.style.fontSize = "12pt";
    }
    return w
}

app.registerExtension({
	name: "cg.sigmas",
	version: 1,
	async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass=="DisplayAnything") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const the_message = message?.[message_id]
                if (the_message) {
                    const w = get_text_widget(this)
                    w.value = the_message.join("")
                    this.onResize?.(this.size)
                }
            }
        }
	},
})