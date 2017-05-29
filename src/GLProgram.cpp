#include "GLProgram.hpp"

std::map<string, GLProgram*> GLProgram::s_programs; // static

inline void checkErrors()
{
	GLenum err = glGetError();
	const char* name;
	switch (err)
	{
	case GL_NO_ERROR:                       name = NULL; break;
	case GL_INVALID_ENUM:                   name = "GL_INVALID_ENUM"; break;
	case GL_INVALID_VALUE:                  name = "GL_INVALID_VALUE"; break;
	case GL_INVALID_OPERATION:              name = "GL_INVALID_OPERATION"; break;
	case GL_STACK_OVERFLOW:                 name = "GL_STACK_OVERFLOW"; break;
	case GL_STACK_UNDERFLOW:                name = "GL_STACK_UNDERFLOW"; break;
	case GL_OUT_OF_MEMORY:                  name = "GL_OUT_OF_MEMORY"; break;
	case GL_INVALID_FRAMEBUFFER_OPERATION:  name = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
	default:                                name = "unknown"; break;
	}

	if (name)
	{
		printf("Caught GL error 0x%04x (%s)!", err, name);
		exit(1);
	}		
}


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
			exit(1);
		}

		std::vector<char> info(infoLen);
		info[0] = '\0';
		glGetShaderInfoLog(shader, infoLen, &infoLen, info.data());
		printf("glCompileShader(%s) failed!\n\n%s", typeStr.c_str(), info.data());
		exit(1);
	}

	checkErrors();
	return shader;
}


void GLProgram::setAttrib(int loc, int size, GLenum type, int stride, GLuint buffer, const void* pointer)
{
	if (loc < 0)
		return;

	glBindBuffer(GL_ARRAY_BUFFER, (buffer) ? buffer : 0);
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, size, type, GL_FALSE, stride, pointer);
	m_numAttribs = std::max(m_numAttribs, loc + 1);
}

void GLProgram::resetAttribs(void)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	for (int i = 0; i < m_numAttribs; i++)
		glDisableVertexAttribArray(i);
	m_numAttribs = 0;
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
			exit(1);
		}

		std::vector<char> info(infoLen);
		info[0] = '\0';
		glGetProgramInfoLog(prog, infoLen, &infoLen, info.data());
		printf("glLinkGLProgram() failed!\n\n%s", info.data());
		exit(1);
	}

	checkErrors();
}


void GLProgram::init(const string& vertexSource, GLenum geomInputType, GLenum geomOutputType, int geomVerticesOut, const string& geometrySource, const string& fragmentSource)
{
	m_glProgram = glCreateProgram();

	// Setup vertex shader
	m_glVertexShader = createGLShader(GL_VERTEX_SHADER, "GL_VERTEX_SHADER", vertexSource);
	glAttachShader(m_glProgram, m_glVertexShader);

	// Setup geometry shader (GL_ARB_geometry_shader4)
	if (geometrySource.length() == 0)
	{
		m_glGeometryShader = 0;
	}
	else
	{
		m_glGeometryShader = createGLShader(GL_GEOMETRY_SHADER_ARB, "GL_GEOMETRY_SHADER_ARB", geometrySource);
		glAttachShader(m_glProgram, m_glGeometryShader);

		if (glProgramParameteriARB == NULL)
		{
			printf("glGLProgramParameteriARB() not available!");
			exit(1);
		}
		glProgramParameteriARB(m_glProgram, GL_GEOMETRY_INPUT_TYPE_ARB, geomInputType);
		glProgramParameteriARB(m_glProgram, GL_GEOMETRY_OUTPUT_TYPE_ARB, geomOutputType);
		glProgramParameteriARB(m_glProgram, GL_GEOMETRY_VERTICES_OUT_ARB, geomVerticesOut);
	}

	// Setup fragment shader
	m_glFragmentShader = createGLShader(GL_FRAGMENT_SHADER, "GL_FRAGMENT_SHADER", fragmentSource);
	glAttachShader(m_glProgram, m_glFragmentShader);

	// Link
	linkGLProgram(m_glProgram);
}