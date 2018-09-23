#pragma once

#include <string>
#include <glad/glad.h>
#include <nanogui/nanogui.h>

using namespace nanogui;

class ProgressView
{
public:
    ProgressView(nanogui::Screen*);
    ~ProgressView() = default;

    // Set function to perform UI update
    void setRenderFunc(std::function<void()> func) { render = func; };

    void showError(const std::string &msg);
    void showMessage(const std::string &primary, float progress);
    void showMessage(const std::string &primary, const std::string &secondary = "");
    void showMessage(const std::string &primary, const std::string &secondary, float progress);
    void center();
    void hide();

private:
    int layout = -1; // detect layout changes
    std::function<void()> render;

    nanogui::Screen *screen; // parent
    nanogui::Window *mProgressWindow;
    nanogui::Label *mProgressLabelPrim;
    nanogui::Label *mProgressLabelSec;
    nanogui::ProgressBar *mProgressBar;
};