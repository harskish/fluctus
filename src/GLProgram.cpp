#include "GLProgram.hpp"
#include "utils.h"

std::map<string, GLProgram*> GLProgram::s_programs; // static


GLProgram::GLProgram(const string& vertexSource, const string& fragmentSource)
{
	init(vertexSource, 0, 0, 0, "", fragmentSource);
}


GLProgram::GLProgram(const string& vertexSource, GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const string& geometrySource, const string& fragmentSource)
{
	init(vertexSource, geomInputType, geomOutputType, geomVerticesOut, geometrySource, fragmentSource);
}


GLProgram::~GLProgram(void)
{
	glDeleteProgram(m_glProgram);
	glDeleteShader(m_glVertexShader);
	glDeleteShader(m_glGeometryShader);
	glDeleteShader(m_glFragmentShader);
    glDeleteVertexArrays(vaos.size(), vaos.data());
}


GLint GLProgram::getAttribLoc(const string& name) const
{
	return glGetAttribLocation(m_glProgram, name.c_str());
}


GLint GLProgram::getUniformLoc(const string& name) const
{
	return glGetUniformLocation(m_glProgram, name.c_str());
}


void GLProgram::use(void)
{
	glUseProgram(m_glProgram);
}

// Insert VAO handles into vector
void GLProgram::addVAOs(GLuint * arr, int num)
{
    for (int i = 0; i < num; i++)
    {
        vaos.push_back(arr[i]);
    }
}

void GLProgram::bindVAO(int ind)
{
    glBindVertexArray(vaos[ind]);
}

// Static
GLProgram* GLProgram::get(const string &name)
{
	auto pos = s_programs.find(name);
	if (pos == s_programs.end()) return NULL;

	return (*pos).second;
}

// Static
void GLProgram::set(const string &name, GLProgram* prog)
{
	GLProgram* old = GLProgram::get(name);
	if (old == prog)
		return;

	if (old)
		delete old;	

	if (prog)
		s_programs[name] = prog;
}

GLuint GLProgram::createGLShader(GLenum type, const string& typeStr, const string& source)
{
	GLuint shader = glCreateShader(type);
	const char* sourcePtr = source.c_str();
	int sourceLen = source.length();
	glShaderSource(shader, 1, &sourcePtr, &sourceLen);
	glCompileShader(shader);

	GLint status = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (!status)
	{
		GLint infoLen = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
		if (!infoLen)
		{
			printf("glCompileShader(%s) failed!", typeStr.c_str());
            waitExit();
		}

		std::vector<char> info(infoLen);
		info[0] = '\0';
		glGetShaderInfoLog(shader, infoLen, &infoLen, info.data());
		printf("glCompileShader(%s) failed!\n\n%s", typeStr.c_str(), info.data());
        waitExit();
	}

	GLcheckErrors();
	return shader;
}


void GLProgram::linkGLProgram(GLuint prog)
{
	glLinkProgram(prog);
	GLint status = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &status);
	if (!status)
	{
		GLint infoLen = 0;
		glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLen);
		if (!infoLen)
		{
			printf("glLinkGLProgram() failed!");
            waitExit();
		}

		std::vector<char> info(infoLen);
		info[0] = '\0';
		glGetProgramInfoLog(prog, infoLen, &infoLen, info.data());
		printf("glLinkGLProgram() failed!\n\n%s", info.data());
        waitExit();
	}

    GLcheckErrors();
}


void GLProgram::init(const string& vertexSource, GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const string& geometrySource, const string& fragmentSource)
{
	m_glProgram = glCreateProgram();

	// Setup vertex shader
	m_glVertexShader = createGLShader(GL_VERTEX_SHADER, "GL_VERTEX_SHADER", vertexSource);
	glAttachShader(m_glProgram, m_glVertexShader);

    GLcheckErrors();

	// Setup geometry shader (GL_ARB_geometry_shader4)
	if (geometrySource.length() == 0)
	{
		m_glGeometryShader = 0;
	}
	else
	{
		m_glGeometryShader = createGLShader(GL_GEOMETRY_SHADER, "GL_GEOMETRY_SHADER", geometrySource);
		glAttachShader(m_glProgram, m_glGeometryShader);
	}

	// Setup fragment shader
	m_glFragmentShader = createGLShader(GL_FRAGMENT_SHADER, "GL_FRAGMENT_SHADER", fragmentSource);
	glAttachShader(m_glProgram, m_glFragmentShader);

	// Link
	linkGLProgram(m_glProgram);
}