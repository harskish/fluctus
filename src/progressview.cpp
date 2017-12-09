#include "progressview.hpp"

ProgressView::ProgressView(nanogui::Screen *parent)
{
    this->screen = parent;
    mProgressWindow = new nanogui::Window(screen, "Please wait");
    mProgressLabelPrim = new nanogui::Label(mProgressWindow, "DEFAULT!");
    mProgressLabelPrim->setFontSize(20);
    mProgressLabelSec = new nanogui::Label(mProgressWindow, "DEF");
    mProgressLabelSec->setFontSize(15);
    mProgressBar = new nanogui::ProgressBar(mProgressWindow);
    mProgressBar->setVisible(true);
    mProgressBar->setFixedWidth(220);
    mProgressWindow->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical, nanogui::Alignment::Minimum, 15, 15));
    mProgressWindow->setFixedWidth(250);
    mProgressWindow->center();
    mProgressWindow->setVisible(false);
    screen->performLayout();
    render = []() { std::cout << "ProgressView: no render function!" << std::endl; };
}

void ProgressView::showError(const std::string & msg)
{
    new nanogui::MessageDialog(screen, nanogui::MessageDialog::Type::Warning, "Error", msg);
    render();
}

// Progress bar replaces secondary text
void ProgressView::showMessage(const std::string & primary, float progress)
{
    if (layout != 0)
    {
        mProgressLabelSec->setVisible(false);
        mProgressBar->setVisible(true);
        mProgressWindow->setVisible(true);
        layout = 0;
    }
    
    mProgressLabelPrim->setCaption(primary);
    mProgressBar->setValue(progress);
    screen->performLayout();
    render();
}

// Progress bar and secondary text visible
void ProgressView::showMessage(const std::string & primary, const std::string & secondary, float progress)
{
    if (layout != 1)
    {
        mProgressLabelSec->setVisible(true);
        mProgressBar->setVisible(true);
        mProgressWindow->setVisible(true);
        layout = 1;
    }

    mProgressLabelPrim->setCaption(primary);
    mProgressLabelSec->setCaption(secondary);
    mProgressBar->setValue(progress);
    screen->performLayout();
    render();
}

void ProgressView::center()
{
    mProgressWindow->center();
}

// Secondary text replaces progress bar
void ProgressView::showMessage(const std::string & primary, const std::string & secondary)
{
    if (layout != 2)
    {
        mProgressLabelSec->setVisible(true);
        mProgressBar->setVisible(false);
        mProgressWindow->setVisible(true);
        layout = 2;
    }

    mProgressLabelPrim->setCaption(primary);
    mProgressLabelSec->setCaption(secondary);
    screen->performLayout();
    render();
}

void ProgressView::hide()
{
    mProgressWindow->setVisible(false);
    layout = -1;
    render();
}



